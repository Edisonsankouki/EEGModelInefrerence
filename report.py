import mne
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
import os
import sys, io
from report_texts import standard_explanation
# ========= NEW: 输出安全化（避免 Windows/GBK stdout 在管道/重定向时崩溃） =========
def sanitize_for_console(text: str, target_encoding: str = None) -> str:
    
    if text is None:
        return ""

    # 目标编码：优先用 stdout 的编码；否则默认 gbk（Windows cmd/pipe 常见）
    enc = (target_encoding or (getattr(sys.stdout, "encoding", None)) or "gbk").lower()

    # 常见危险字符替换（可按需要增删）
    # 说明：用 translate 做单字符替换；多字符（如 'μV²'）会自然变为 'uV^2'
    trans = str.maketrans({
        "μ": "u",
        "²": "^2",
        "³": "^3",
        "α": "alpha",
        "β": "beta",
        "θ": "theta",
        "≥": ">=",
        "≤": "<=",
        "≠": "!=",
        "–": "-",     # en dash
        "—": "-",     # em dash
        "→": "->",
        "×": "x",
        "…": "...",
        "Ⅰ": "I",
        "Ⅱ": "II",
        "Ⅲ": "III",
        "Ⅳ": "IV",
        "Ⅴ": "V",
        "·": ".",
        "•": "*",
        "\u00a0": " ",  # nbsp
    })

    text2 = str(text).translate(trans)

    # 兜底：如果仍无法用目标编码编码，则把不可编码字符转义成 \uXXXX（ASCII 安全）
    try:
        text2.encode(enc)
        return text2
    except UnicodeEncodeError:
        return text2.encode(enc, errors="backslashreplace").decode(enc, errors="strict")

# ===== 通道不匹配：统一异常与花括号打印 =====
class ChannelMismatchError(Exception):
    def __init__(self, msg, details=None):
        super().__init__(msg)
        self.details = details or {}

def brace_print_dict(d):
    print("{")
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            vs = ", ".join([str(x) for x in v]) if v else ""
            print(f"  {k}: [{vs}]")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                if isinstance(v2, (list, tuple)):
                    vs2 = ", ".join([str(x) for x in v2]) if v2 else ""
                    print(f"    {k2}: [{vs2}]")
                else:
                    print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")
    print("}")

# ========= 通用评级/文本工具 =========
def clamp_score(x, lo=0.0, hi=100.0):
    try:
        v = float(x)
    except Exception:
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return float(min(max(v, lo), hi))

def level_from_score(score):
    """
    分段：I(0-85), II(85-90), III(90-95), IV(95-100)
    采用：s < 85 -> I；s < 90 -> II；s < 95 -> III；s >= 95 -> IV
    （边界进入更高一级：85→II，90→III，95→IV）
    """
    if score is None:
        return "NA", "无法评估"
    s = clamp_score(score)
    if s is None:
        return "NA", "无法评估"

    if s < 85:
        return "I", "正常"
    elif s < 90:
        return "II", "关注"
    elif s < 95:
        return "III", "预警"
    else:
        return "IV", "高危"




def make_metric_record(name, score, level=None, level_name=None, brief=""):
    s = clamp_score(score) if score is not None else None
    if level is None or level_name is None:
        level, level_name = level_from_score(s)
    return {
        "name": name,
        "score": s,
        "level": level,
        "level_name": level_name,
        "brief": brief
    }

def format_metric_line(m):
    if m["level"] == "NA":
        return f"- {m['name']}: NA\n{m.get('brief','').strip()}"
    sc = "NA" if m["score"] is None else f"{m['score']:.1f}"
    # 注意：不在这里改写解释文本，解释文本本身已含【Ⅰ级】等
    return f"- {m['name']}: 风险分={sc}，等级={m['level']}（{m['level_name']}）\n{m.get('brief','').strip()}"

def _robust_aggregate(scores, method="top2_mean"):
    """
    更保守的模块汇总，避免单项偏高把模块拉到高风险。
    method:
      - "top2_mean": 取最高的两个分数求均值（比 max 保守，但仍关注风险）
      - "p75": 取 75 分位（更平滑）
      - "mean": 普通均值（最保守，但可能掩盖单点异常）
    """
    scores = [float(s) for s in scores if s is not None and np.isfinite(s)]
    if not scores:
        return None
    scores = sorted(scores)
    if method == "mean":
        return float(np.mean(scores))
    if method == "p75":
        return float(np.percentile(scores, 75))
    # default: top2_mean
    if len(scores) == 1:
        return float(scores[0])
    return float(np.mean(scores[-2:]))

def module_summary(metrics, method="top2_mean", require_multi_hits=True):
    """
    模块汇总（保守版）：
      1) 用鲁棒聚合替代 max
      2) 可选：要求“至少两项>=II段”才把模块提升到 II/III（防止单项误报）
    """
    valid = [m for m in metrics if m.get("score") is not None]
    if not valid:
        return {"module_score": None, "module_level": "NA", "module_level_name": "无法评估", "module_brief": "无法评估（缺少可用代理指标）"}

    scores = [m["score"] for m in valid]
    agg = _robust_aggregate(scores, method=method)

    lvl, lvl_name = level_from_score(agg)

    if require_multi_hits:
        # 至少两项进入 II(>65) 才把模块提升到 II；否则保持 I（避免单点拉高）
        hits_ii = sum(1 for s in scores if s is not None and s > 85)
        hits_iii = sum(1 for s in scores if s is not None and s > 90)
        hits_iv = sum(1 for s in scores if s is not None and s > 95)

        # 如果只有 0~1 项偏高，模块不升级
        if lvl in ("II", "III", "IV") and hits_ii < 2:
            lvl, lvl_name = "I", "正常"

        # III/IV 同理要求更强一致性
        if lvl == "III" and hits_iii < 2:
            lvl, lvl_name = "II", "关注"
        if lvl == "IV" and hits_iv < 2:
            lvl, lvl_name = "III", "预警"

    # 风险点用 top1 提示即可
    max_m = max(valid, key=lambda x: x["score"])
    brief = f"模块整体为{lvl_name}，主要风险点：{max_m['name']}（模块得分={max_m['score']:.1f}）。"
    return {"module_score": agg, "module_level": lvl, "module_level_name": lvl_name, "module_brief": brief}




# ========= 工具函数 =========
def compute_snr(signal, fs):
    """Welch 估计下的 SNR（1–30 Hz 为信号，其余为噪声）"""
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    signal_freq_mask = (freqs >= 0.5) & (freqs <= 30)
    noise_freq_mask = ~signal_freq_mask
    signal_power = np.trapz(psd[signal_freq_mask], freqs[signal_freq_mask])
    noise_power = np.trapz(psd[noise_freq_mask], freqs[noise_freq_mask])
    snr = 10 * np.log10(signal_power / max(noise_power, 1e-12))
    return snr

def compute_psd(signal, fs):
    """Welch 功率谱密度"""
    freqs, psd = welch(signal, fs=fs, nperseg=1024)
    return freqs, psd

def amplitude_descriptor(amplitude_uv):
    """按 RMS 波幅（μV）给出定性描述"""
    a = amplitude_uv
    if a < 10:
        return "低波幅的"
    elif a < 20:
        return "低-中波幅的"
    elif a < 30:
        return "低-高波幅的"
    elif a < 40:
        return "中-高波幅的"
    else:
        return "高波幅的"

def compute_dominant_band_global_peak(freqs, psd, band_range):
    """
    在 band_range 内找全局峰，向两侧扩展到峰值50%的位置，返回该频率区间边界。
    """
    idx = np.where((freqs >= band_range[0]) & (freqs <= band_range[1]))[0]
    if len(idx) == 0:
        return None, None
    sub_freqs = freqs[idx]
    sub_psd = psd[idx]
    peak_idx = np.argmax(sub_psd)
    peak_psd = sub_psd[peak_idx]
    left_idx = peak_idx
    while left_idx > 0 and sub_psd[left_idx] > 0.5 * peak_psd:
        left_idx -= 1
    right_idx = peak_idx
    while right_idx < len(sub_freqs) - 1 and sub_psd[right_idx] > 0.5 * peak_psd:
        right_idx += 1
    low_bound = sub_freqs[left_idx]
    high_bound = sub_freqs[right_idx]
    return low_bound, high_bound

# ========= 读数与窗口选择 =========
def read_edf(edf_file, window_length_sec=10):
    """
    返回:
      best_window_eeg: (n_ch, win_len)
      best_window_eeg_O1O2: (2, win_len) 或 None
      fs: 采样率
      channel_positions: dict[idx] = (x,y)
      raw: MNE Raw（已标准命名/设置montage）
    若标准通道命中不足阈值，抛出 ChannelMismatchError 并附详细原因。
    """
    import traceback

    try:
        raw = mne.io.read_raw_edf(edf_file, preload=True, encoding='latin1', verbose=False)
    except Exception as e:
        raise ChannelMismatchError("EDF 文件读取失败", {"exception": repr(e)})

    # --- 目标标准通道 & 常见别名（大小写不敏感；覆盖 T3/T4/T5/T6 ↔ T7/T8/P7/P8）
    TARGETS = [
        'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
        'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4',
        'T6', 'O1', 'O2', 'Oz'
    ]
    ALIASES = {
        'T3': ['T3','T7'], 'T4': ['T4','T8'], 'T5': ['T5','P7'], 'T6': ['T6','P8'],
        'Fp1':['Fp1','FP1'], 'Fp2':['Fp2','FP2'],
        'F7':['F7'], 'F3':['F3'], 'Fz':['Fz','FZ'], 'F4':['F4'], 'F8':['F8'],
        'C3':['C3'], 'Cz':['Cz','CZ'], 'C4':['C4'],
        'P3':['P3'], 'Pz':['Pz','PZ'], 'P4':['P4'],
        'O1':['O1'], 'O2':['O2'], 'Oz':['Oz','OZ'],
    }

    all_chs = raw.ch_names

    def candidates_for(std):
        cands = []
        for alias in ALIASES.get(std, [std]):
            al = alias.lower()
            hits = [name for name in all_chs if al in name.lower()]
            cands.extend(hits)
        # 去重保序
        return list(dict.fromkeys(cands))

    def score(name, std):
        n = name.lower()
        s = 0
        # 更偏好 -REF 或 * REF 结尾
        if "-ref" in n or n.endswith(" ref") or n.endswith("- ref"):
            s += 3
        # 命中别名本体再加分
        if any(alias in name for alias in ALIASES.get(std, [std])):
            s += 1
        # 词首/词尾轻微加分
        base = std.lower()
        if n.startswith(base) or n.endswith(base):
            s += 1
        return -s  # 升序排序，所以负号

    chosen = {}        # {std: matched_name}
    ambiguous = {}     # {std: [c1,c2,...]}
    missing = []       # [std,...]

    for std in TARGETS:
        cand = candidates_for(std)
        if not cand:
            missing.append(std)
            continue
        cand_sorted = sorted(cand, key=lambda nm: score(nm, std))
        chosen[std] = cand_sorted[0]
        if len(cand_sorted) > 1:
            ambiguous[std] = cand_sorted[:8]

    # === 阈值校验：至少命中 8 个标准通道，否则认为“通道不匹配”
    if len(chosen) < 8:
        raise ChannelMismatchError(
            "通道不匹配：可用标准通道 < 8，无法稳定生成频谱与对称/调节评估",
            {
                "已匹配映射数": len(chosen),
                "缺失关键通道": missing if missing else "无",
                "歧义匹配": ambiguous if ambiguous else "无",
                "文件包含通道数": len(all_chs),
                "文件示例通道": all_chs[:20],
                "提示": "建议确保至少 8 个标准 10-20 通道；T3/T4/T5/T6 也可命名为 T7/T8/P7/P8；优先含 -REF。",
            }
        )

    # === 重命名为标准标签（例如把 “EEG FP1-REF” 改成 “Fp1”）
    rename_map = {orig: std for std, orig in chosen.items()}
    raw.rename_channels(rename_map)

    # === 只挑出已匹配标准通道（避免噪声通道干扰）
    raw.pick(list(rename_map.values()))

    # notch 与 montage（容错）
    try:
        raw.notch_filter(freqs=50, method='spectrum_fit', filter_length='auto', verbose=False)
    except Exception:
        pass

    try:
        if hasattr(sys, "_MEIPASS"):
            montage_path = os.path.join(sys._MEIPASS, "mne", "channels", "data", "montages")
            if os.path.exists(montage_path):
                os.environ["MNE_DATA"] = montage_path
        montage = mne.channels.make_standard_montage('standard_1020')
        raw.set_montage(montage)
    except Exception:
        # montage 失败不致命
        pass

    eeg_data = raw.get_data()
    fs = float(raw.info["sfreq"])

    # 通道坐标（可能为 0.0,0.0，如果 montage 未成功）
    channel_positions = {}
    for idx, ch in enumerate(raw.info['chs']):
        try:
            x, y = ch['loc'][0], ch['loc'][1]
        except Exception:
            x, y = 0.0, 0.0
        channel_positions[idx] = (x, y)

    # === SNR 最佳窗口（10s，步长 0.5s）
    window_length = int(window_length_sec * fs)
    step_length = int(0.5 * fs)
    num_samples = eeg_data.shape[1]
    num_windows = max(0, (num_samples - window_length) // step_length + 1)

    if num_windows <= 0:
        raise ChannelMismatchError("有效数据过短，无法切出 1 个 10s 窗口", {
            "采样率": fs, "总点数": int(raw.n_times), "窗口点数": window_length
        })

    max_snr_eeg = -np.inf
    best_window_eeg, best_start_idx = None, 0
    for i in range(num_windows):
        start_idx = i * step_length
        window_eeg = eeg_data[:, start_idx:start_idx + window_length]
        snr_eeg = np.mean([compute_snr(window_eeg[j], fs) for j in range(window_eeg.shape[0])])
        if snr_eeg > max_snr_eeg:
            max_snr_eeg = snr_eeg
            best_window_eeg = window_eeg
            best_start_idx = start_idx

    # === O1/O2 同步窗口（若存在）
    best_window_eeg_O1O2 = None
    have_o1, have_o2 = ('O1' in raw.ch_names), ('O2' in raw.ch_names)
    if best_window_eeg is not None and (have_o1 or have_o2):
        want = [c for c in ('O1','O2') if c in raw.ch_names]
        try:
            raw_o = raw.copy().pick(want)
            data_o = raw_o.get_data()[:, best_start_idx:best_start_idx + window_length]
            best_window_eeg_O1O2 = data_o
        except Exception:
            pass

    return best_window_eeg, best_window_eeg_O1O2, fs, channel_positions, raw

# ========= NEW: 对称性判定（成对导联，30%阈值） =========
PAIR_LIST = [
    ('O1', 'O2'), ('P3', 'P4'), ('C3', 'C4'),
    ('F3', 'F4'), ('Fp1', 'Fp2'), ('T3', 'T4'), ('T5', 'T6')
]

def rms_uv(x):
    return float(np.sqrt(np.mean(np.square(x)))) * 1e6  # μV

def check_symmetry_by_pairs(raw, window_data):
    """
    按标准：|(A-B)| / max(A,B) < 0.3 即为对称。
    使用 0.5–30 Hz 带通后的 RMS（μV）作为“波幅”。
    返回 (pair_results, overall_text)
    """
    fs = raw.info['sfreq']
    filtered = mne.filter.filter_data(window_data.copy(), fs, l_freq=0.5, h_freq=30, verbose=False)

    name_to_idx = {name: i for i, name in enumerate(raw.ch_names)}
    pair_results = []
    asym_flags = []

    for a, b in PAIR_LIST:
        if a in name_to_idx and b in name_to_idx:
            ia, ib = name_to_idx[a], name_to_idx[b]
            amp_a = rms_uv(filtered[ia])
            amp_b = rms_uv(filtered[ib])
            m = max(amp_a, amp_b, 1e-9)
            ratio = abs(amp_a - amp_b) / m
            ok = ratio < 0.3
            pair_results.append({
                "pair": f"{a}-{b}",
                "amp_uv": (amp_a, amp_b),
                "ratio": ratio,
                "symmetric": ok
            })
            asym_flags.append(ok)
        else:
            pair_results.append({
                "pair": f"{a}-{b}",
                "amp_uv": (None, None),
                "ratio": None,
                "symmetric": None,
                "note": "缺少通道"
            })

    if any(v is False for v in asym_flags):
        overall = "左右不对称（存在 >=30% 波幅差的相应导联）"
    elif all(v is True for v in asym_flags) and len(asym_flags) > 0:
        overall = "左右对称（所有成对导联差 <30%）"
    else:
        overall = "对称性无法完整评估（成对通道缺失）"

    return pair_results, overall

# ========= NEW: 调节性判定（仅 O1/O2，两个条件） =========
def _dominant_freq_hz(segment, fs):
    """用 Welch 在 4–30 Hz 范围内求主频（α-快波为主，更抗工频/超慢波干扰）"""
    f, p = welch(segment, fs=fs, nperseg=min(1024, len(segment)))
    mask = (f >= 4) & (f <= 30)
    if not np.any(mask):
        return np.nan
    f_sel = f[mask]
    p_sel = p[mask]
    return float(f_sel[np.argmax(p_sel)])

def check_regulation_o1o2(o1, o2, fs, segment_len_sec=3.0):
    """
    标准：
      条件①（单导联内）：同一导联分段（3s），若跨段主频的极差 > 1 Hz（O1 或 O2 任一满足）=> 调节不良
      条件②（O1 vs O2）：对齐分段后，若某段 |f_O1 - f_O2| > 0.5 Hz => 调节不良
    返回 (detail_dict, overall_text)
    """
    n = len(o1)
    L = int(segment_len_sec * fs)
    k = n // L
    if k == 0:
        return {"note": "数据过短，无法按 3s 分段"}, "调节无法评估（数据过短）"

    o1 = o1[:k*L]
    o2 = o2[:k*L]

    f1, f2 = [], []
    for i in range(k):
        s = slice(i*L, (i+1)*L)
        f1.append(_dominant_freq_hz(o1[s], fs))
        f2.append(_dominant_freq_hz(o2[s], fs))
    f1 = np.array(f1)
    f2 = np.array(f2)

    cond1_o1 = (np.nanmax(f1) - np.nanmin(f1)) > 1.0 if np.all(np.isfinite(f1)) else False
    cond1_o2 = (np.nanmax(f2) - np.nanmin(f2)) > 1.0 if np.all(np.isfinite(f2)) else False
    cond1 = cond1_o1 or cond1_o2

    if np.all(np.isfinite(f1)) and np.all(np.isfinite(f2)):
        diffs = np.abs(f1 - f2)
        cond2 = np.any(diffs > 0.5)
    else:
        diffs = None
        cond2 = False

    bad = cond1 or cond2
    overall = "调节不良" if bad else "调节正常"

    detail = {
        "segment_len_sec": segment_len_sec,
        "f_o1_hz": f1.tolist(),
        "f_o2_hz": f2.tolist(),
        "cond1_o1_range_hz": (None if not np.any(np.isfinite(f1)) else float(np.nanmax(f1) - np.nanmin(f1))),
        "cond1_o2_range_hz": (None if not np.any(np.isfinite(f2)) else float(np.nanmax(f2) - np.nanmin(f2))),
        "cond1_trigger": bool(cond1),
        "diffs_abs_hz": (None if diffs is None else diffs.tolist()),
        "cond2_trigger": bool(cond2),
    }
    return detail, overall

# ========= NEW: 功能检测（基于 EEG 频谱/对称/噪声 的代理指标 -> 固定中文结论 + 结构化评级） =========
def bandpower_from_psd(freqs, psd, band):
    """psd: (..., F) 频谱功率，返回 band 积分功率"""
    lo, hi = band
    m = (freqs >= lo) & (freqs <= hi)
    if not np.any(m):
        return 0.0
    return float(np.trapz(psd[..., m], freqs[m]))

def compute_multich_psd(window_data, fs, nperseg=1024):
    """
    window_data: (n_ch, n_samples)
    返回 freqs, psd_ch: (n_ch, F)
    """
    psd_ch = []
    freqs = None
    for ch in range(window_data.shape[0]):
        f, p = welch(window_data[ch], fs=fs, nperseg=min(nperseg, window_data.shape[1]))
        freqs = f if freqs is None else freqs
        psd_ch.append(p)
    return freqs, np.stack(psd_ch, axis=0)

def spectral_entropy(psd, eps=1e-12):
    """psd: (F,) 归一化谱熵（0~lnF）"""
    p = psd.astype(np.float64)
    p = p / (np.sum(p) + eps)
    return float(-np.sum(p * np.log(p + eps)))

def safe_log(x, eps=1e-12):
    return float(np.log(max(float(x), eps)))
def shrink_toward(x, center=50.0, factor=1.0):
    """
    将分数向 center 收缩：factor<1 更保守，factor=1 不变。
    例如 x=90, center=50, factor=0.7 -> 50 + (40*0.7)=78
    """
    if x is None:
        return None
    x = float(x)
    if not np.isfinite(x):
        return None
    return clamp_score(center + (x - center) * float(factor))

def _linear_map(x, x0, x1, y0, y1):
    """x in [x0,x1] -> y in [y0,y1], outside clipped"""
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    t = min(max(float(t), 0.0), 1.0)
    return float(y0 + t * (y1 - y0))

def _score_fatigue(fatigue_index):
    """
    fatigue_index = theta_alpha_ratio / beta_alpha_ratio
    低=好，高=差；输出 0-100 风险分（越高越差）
    保守化：轻度升高不要直接推到 80+
    """
    if fatigue_index is None or np.isnan(fatigue_index) or np.isinf(fatigue_index):
        return None
    fi = float(fatigue_index)

    # 更保守的阈值（你可按数据再校准）
    if fi <= 1.35:
        return 48.0
    elif fi <= 2.10:
        return _linear_map(fi, 1.35, 2.10, 48.0, 70.0)   # 到 70 才算“关注上沿”
    elif fi <= 2.80:
        return _linear_map(fi, 2.10, 2.80, 70.0, 85.0)
    elif fi <= 3.40:
        return _linear_map(fi, 2.80, 3.40, 85.0, 95.0)
    else:
        return 98.0

def _score_noise(line_ratio, emg_ratio):
    """
    工频/肌电污染越高，风险越高。返回 0-100 风险分（越高越差）
    """
    if line_ratio is None or emg_ratio is None:
        return None
    lr = float(line_ratio)
    er = float(emg_ratio)
    # 分量：低于阈值不加分，超过阈值逐步增加
    line_comp = 0.0
    if lr <= 0.08:
        line_comp = 0.0
    elif lr <= 0.15:
        line_comp = _linear_map(lr, 0.08, 0.15, 0.0, 15.0)
    else:
        line_comp = 15.0 + _linear_map(min(lr, 0.30), 0.15, 0.30, 0.0, 20.0)  # 最多 35

    emg_comp = 0.0
    if er <= 0.20:
        emg_comp = 0.0
    elif er <= 0.35:
        emg_comp = _linear_map(er, 0.20, 0.35, 0.0, 15.0)
    else:
        emg_comp = 15.0 + _linear_map(min(er, 0.60), 0.35, 0.60, 0.0, 20.0)  # 最多 35

    base = 45.0
    return clamp_score(base + line_comp + emg_comp)

def _score_stability(line_ratio, emg_ratio, regulation_ok):
    s = _score_noise(line_ratio, emg_ratio)
    if s is None:
        return None
    # 调节不良显著加重
    if regulation_ok is False:
        s += 20.0
    return clamp_score(s)

def _score_flex(spec_ent, rel_alpha):
    """
    spec_ent 过低可能提示节律单一/伪迹/僵化；此处做保守代理映射
    （已校准v2：评分上限控制在正常范围，与eeg_batch_analysis一致）
    """
    if spec_ent is None or np.isnan(spec_ent) or np.isinf(spec_ent):
        return None
    se = float(spec_ent)
    # 校准v2：整体下移约13分，最大82+3=85
    if se >= 4.5:
        score = 42.0
    elif se >= 4.0:
        score = _linear_map(se, 4.0, 4.5, 55.0, 42.0)
    elif se >= 3.5:
        score = _linear_map(se, 3.5, 4.0, 72.0, 55.0)
    else:
        score = 82.0

    # alpha 占比极端也加一点风险（减弱：10→3）
    if rel_alpha is not None:
        ra = float(rel_alpha)
        if ra < 0.05 or ra > 0.40:
            score += 3.0
    return clamp_score(score)

def _score_reaction(beta_alpha_ratio):
    """
    静息 EEG 无法直接测反应时；用 beta/alpha 做保守代理，默认中性
    （已校准v2：评分上限控制在正常范围，与eeg_batch_analysis一致）
    """
    if beta_alpha_ratio is None or np.isnan(beta_alpha_ratio) or np.isinf(beta_alpha_ratio):
        return None
    bar = float(beta_alpha_ratio)
    score = 55.0
    if bar > 2.0:
        score += 20.0
    elif bar > 1.6:
        score += 10.0
    elif bar < 0.4:
        score += 15.0
    elif bar < 0.6:
        score += 7.0
    return clamp_score(score)

def _score_arousal(beta_alpha_ratio, faa_abs):
    """
    高唤醒/应激代理分（保守版）
    """
    score = 45.0

    if beta_alpha_ratio is not None:
        bar = float(beta_alpha_ratio)
        if bar > 2.2:
            score += 20.0
        elif bar > 1.8:
            score += 12.0
        elif bar > 1.5:
            score += 6.0
        # 低 beta/alpha 不直接推高唤醒风险
    if faa_abs is not None:
        fa = float(faa_abs)
        if fa > 0.90:
            score += 12.0
        elif fa > 0.75:
            score += 7.0
        elif fa > 0.60:
            score += 4.0

    return clamp_score(score)


def _score_depression(theta_alpha_ratio, faa_abs, fatigue_score):
    """
    抑郁：默认低风险（I/II），只有慢化+额叶不对称+疲劳共同明显时才升到 III/IV
    """
    score = 45.0

    if theta_alpha_ratio is not None:
        tar = float(theta_alpha_ratio)
        if tar > 2.0:
            score += 18.0
        elif tar > 1.6:
            score += 10.0
        elif tar > 1.3:
            score += 5.0

    if faa_abs is not None:
        fa = float(faa_abs)
        if fa > 0.90:
            score += 10.0
        elif fa > 0.75:
            score += 6.0

    # 只有疲劳非常高才额外加
    if fatigue_score is not None and fatigue_score > 90:
        score += 6.0

    return clamp_score(score)


def compute_functional_results(raw, window_data, sfreq, pair_results=None, reg_detail=None, line_freq=50):
    """
    输出:
      - result_dict: 结构化指标（包含 ratings/module_summary）
      - result_text: 报告文本（含原三段结论 + 新增结构化评级板块）
    """
    fs = float(sfreq)
    freqs, psd_ch = compute_multich_psd(window_data, fs, nperseg=1024)  # (n_ch, F)

    # --- 频带定义（慢波/alpha/快波 + theta/beta 用于若干指数）
    band_total = (0.5, 45)
    band_theta = (4, 8)
    band_alpha = (8, 13)
    band_beta  = (13, 30)
    band_emg   = (30, 45)
    band_line  = (line_freq - 1, line_freq + 1)

    psd_avg = psd_ch.mean(axis=0)

    total_p = bandpower_from_psd(freqs, psd_avg, band_total)
    theta_p = bandpower_from_psd(freqs, psd_avg, band_theta)
    alpha_p = bandpower_from_psd(freqs, psd_avg, band_alpha)
    beta_p  = bandpower_from_psd(freqs, psd_avg, band_beta)
    emg_p   = bandpower_from_psd(freqs, psd_avg, band_emg)
    line_p  = bandpower_from_psd(freqs, psd_avg, band_line)

    rel_alpha = alpha_p / max(total_p, 1e-12)
    rel_beta  = beta_p  / max(total_p, 1e-12)
    rel_theta = theta_p / max(total_p, 1e-12)

    theta_alpha_ratio = theta_p / max(alpha_p, 1e-12)
    beta_alpha_ratio  = beta_p  / max(alpha_p, 1e-12)
    emg_ratio         = emg_p   / max(total_p, 1e-12)
    line_ratio        = line_p  / max(total_p, 1e-12)
    spec_ent          = spectral_entropy(psd_avg[(freqs >= 0.5) & (freqs <= 45)])

    # --- 对称性：成对导联 30%标准
    symmetry_ok = None
    if pair_results is not None:
        flags = [it.get("symmetric") for it in pair_results if it.get("symmetric") is not None]
        if len(flags) > 0:
            symmetry_ok = all(flags)

    # --- 调节性：O1/O2
    regulation_ok = None
    if isinstance(reg_detail, dict) and ("cond1_trigger" in reg_detail) and ("cond2_trigger" in reg_detail):
        regulation_ok = (not reg_detail["cond1_trigger"]) and (not reg_detail["cond2_trigger"])

    # --- 额叶 alpha 不对称（FAA）proxy：F3/F4 alpha 功率 log 差
    name_to_idx = {name: i for i, name in enumerate(raw.ch_names)}
    faa = None
    if ("F3" in name_to_idx) and ("F4" in name_to_idx):
        i3, i4 = name_to_idx["F3"], name_to_idx["F4"]
        alpha_f3 = bandpower_from_psd(freqs, psd_ch[i3], band_alpha)
        alpha_f4 = bandpower_from_psd(freqs, psd_ch[i4], band_alpha)
        faa = safe_log(alpha_f4) - safe_log(alpha_f3)
    faa_abs = None if faa is None else abs(faa)

    # ========= 原有“固定句式”结论（保留现有功能，不破坏） =========
    if regulation_ok is True and (line_ratio < 0.08) and (emg_ratio < 0.20):
        brain_stability = "正常"
    elif regulation_ok is False or (line_ratio > 0.15) or (emg_ratio > 0.35):
        brain_stability = "中等"
    else:
        brain_stability = "正常"

    if (line_ratio < 0.08) and (emg_ratio < 0.20):
        anti_interf = "正常"
    elif (line_ratio > 0.15) or (emg_ratio > 0.35):
        anti_interf = "中等"
    else:
        anti_interf = "正常"

    if (spec_ent >= 4.5) and (rel_alpha >= 0.08) and (rel_alpha <= 0.35):
        flex_pattern = "正常"
    else:
        flex_pattern = "正常"

    if beta_alpha_ratio > 1.2:
        reaction_speed = "中等"
    elif beta_alpha_ratio < 0.6:
        reaction_speed = "中等"
    else:
        reaction_speed = "中等"

    if (symmetry_ok is True) and (faa_abs is None or faa_abs < 0.35):
        lateralization = "未见显著左右脑偏侧"
    elif symmetry_ok is False or (faa_abs is not None and faa_abs >= 0.60):
        lateralization = "存在一定左右差异（建议结合临床/任务进一步评估）"
    else:
        lateralization = "未见显著左右脑偏侧"

    if theta_alpha_ratio < 0.9:
        ad_proxy = "风险信号较低（慢化指标未见显著升高）"
    elif theta_alpha_ratio > 1.4:
        ad_proxy = "风险信号中等（慢化指标偏高，需结合量表/影像进一步判断）"
    else:
        ad_proxy = "风险信号较低（慢化指标未见显著升高）"

    vitality = "中等"  # 保留原有口径（文本）

    fatigue_index = (theta_alpha_ratio / max(beta_alpha_ratio, 1e-12))
    # 这里仍保留原有“正常/中等”输出，但我们会用统一 fatigue_score 来保证一致性
    if fatigue_index < 1.2:
        brain_fatigue = "正常"
    elif fatigue_index > 2.0:
        brain_fatigue = "中等"
    else:
        brain_fatigue = "正常"

    oxygen_proxy = "正常"
    spo2_proxy   = "正常"

    high_arousal = (beta_alpha_ratio > 1.6) or (faa_abs is not None and faa_abs > 0.75)
    if high_arousal:
        mental_text = "未见显著恐惧、易怒、自闭、疲劳、纠结或安全感缺失（EEG 代理指标未提示明显高唤醒/应激异常），抗挫能力中等。"
    else:
        mental_text = "未见显著恐惧、易怒、自闭、疲劳、纠结或安全感缺失，抗挫能力中等。"

    cog_text = (
        f"认知功能：脑稳定度{brain_stability}，抵抗干扰{anti_interf}，思维弹性与模式{flex_pattern}，"
        f"行为反应速度{reaction_speed}，{lateralization}，认知减退相关风险信号{ad_proxy}。"
    )
    phys_text = (
        f"脑生理监测：脑耗氧水平{oxygen_proxy}，脑疲劳水平{brain_fatigue}，"
        f"血氧代谢水平{spo2_proxy}，大脑活力{vitality}。"
    )
    psy_text = f"心理健康：{mental_text}"


    anti_interf_score = shrink_toward(_score_noise(line_ratio, emg_ratio), center=50.0, factor=1.0)  # 干扰本身不收缩
        # ========= 数据质量抑制：噪声越大，越不应报高风险（向 50 收缩） =========
    # anti_interf_score 越高表示工频/肌电污染越大
    if anti_interf_score is None:
        quality_factor = 1.0
    else:
        if anti_interf_score >= 95:
            quality_factor = 0.55
        elif anti_interf_score >= 85:
            quality_factor = 0.70
        elif anti_interf_score >= 75:
            quality_factor = 0.85
        else:
            quality_factor = 1.0
    
    fatigue_score = shrink_toward(_score_fatigue(fatigue_index), center=50.0, factor=quality_factor)
    stability_score   = shrink_toward(_score_stability(line_ratio, emg_ratio, regulation_ok), center=50.0, factor=quality_factor)
    flex_score        = shrink_toward(_score_flex(spec_ent, rel_alpha), center=50.0, factor=quality_factor)
    reaction_score    = shrink_toward(_score_reaction(beta_alpha_ratio), center=50.0, factor=quality_factor)

    arousal_score     = shrink_toward(_score_arousal(beta_alpha_ratio, faa_abs), center=50.0, factor=quality_factor)


    base_psy = 45.0

    security_score = clamp_score(
        base_psy
        + (arousal_score - 50.0) * 0.35
        + ((fatigue_score - 50.0) * 0.20 if fatigue_score is not None else 0.0)
    )
    resilience_score = clamp_score(
        base_psy
        + ((fatigue_score - 50.0) * 0.30 if fatigue_score is not None else 0.0)
        + (arousal_score - 50.0) * 0.15
    )

    depression_score = shrink_toward(_score_depression(theta_alpha_ratio, faa_abs, fatigue_score), center=50.0, factor=quality_factor)

    indecision_score = clamp_score(
        base_psy
        + (arousal_score - 50.0) * 0.25
        + ((fatigue_score - 50.0) * 0.20 if fatigue_score is not None else 0.0)
        + ((flex_score - 50.0) * 0.15 if flex_score is not None else 0.0)
    )

    fear_score = clamp_score(40.0 + (arousal_score - 50.0) * 0.28)  # 校准v2：基础分45→40，系数0.45→0.28
    irritability_score = clamp_score(
        base_psy
        + (arousal_score - 50.0) * 0.45
        + ((fatigue_score - 50.0) * 0.15 if fatigue_score is not None else 0.0)
    )
    social_withdrawal_score = clamp_score(
        base_psy
        + ((fatigue_score - 50.0) * 0.25 if fatigue_score is not None else 0.0)
        + (arousal_score - 50.0) * 0.10
    )

    # 最后整体再做一次质量收缩（关键：噪声大时别报高风险）
    security_score = shrink_toward(security_score, center=50.0, factor=quality_factor)
    resilience_score = shrink_toward(resilience_score, center=50.0, factor=quality_factor)
    indecision_score = shrink_toward(indecision_score, center=50.0, factor=quality_factor)
    fear_score = shrink_toward(fear_score, center=50.0, factor=quality_factor)
    irritability_score = shrink_toward(irritability_score, center=50.0, factor=quality_factor)
    social_withdrawal_score = shrink_toward(social_withdrawal_score, center=50.0, factor=quality_factor)

    # 脑耗氧/血氧代谢/活力（EEG代理口径）
    # 这里保持保守：不做“诊断式”分布；用疲劳与噪声轻度驱动
    oxygen_score = clamp_score(55.0 + (0.3 * (fatigue_score - 50.0) if fatigue_score is not None else 0.0) + (0.2 * ((anti_interf_score or 55.0) - 55.0)))
    spo2_score   = clamp_score(45.0 + (0.15 * (fatigue_score - 50.0) if fatigue_score is not None else 0.0) + (0.12 * ((anti_interf_score or 55.0) - 55.0)))  # 校准v2：基础55→45，系数0.2→0.15/0.12

    # 活力：rel_beta 过低更差（风险分更高），过高不做过度推断
    vitality_score = None
    if rel_beta is not None and np.isfinite(rel_beta):
        rb = float(rel_beta)
        # 校准v2：整体下移约16分，最大62+2=64
        if rb < 0.10:
            vitality_score = 62.0
        elif rb < 0.14:
            vitality_score = _linear_map(rb, 0.10, 0.14, 62.0, 52.0)
        elif rb < 0.22:
            vitality_score = _linear_map(rb, 0.14, 0.22, 52.0, 45.0)
        else:
            vitality_score = 45.0
        # 疲劳高时略上调（减弱：5→2）
        if fatigue_score is not None and fatigue_score > 85:
            vitality_score += 2.0
    vitality_score = clamp_score(vitality_score) if vitality_score is not None else None

    # 睡眠质量：用疲劳代理推断（仅两类：正常/较差），但仍输出四级风险分
    sleep_score = None
    if fatigue_score is not None:
        if fatigue_score <= 65:
            sleep_score = 55.0
        elif fatigue_score <= 85:
            sleep_score = 75.0
        elif fatigue_score <= 95:
            sleep_score = 90.0
        else:
            sleep_score = 97.0
    sleep_level, sleep_level_name = level_from_score(sleep_score)

    sleep_quality_text = "睡眠质量正常" if (sleep_score is not None and sleep_score <= 65) else "睡眠质量较差"
    # 注意：睡眠模块的报告“状态文本”与“分级”会并存

    # ========= 组装 ratings（所有指标必须输出） =========
    ratings_psy = []
    for nm, sc in [
        ("安全感", security_score),
        ("心理韧性", resilience_score),
      #  ("抑郁水平", depression_score),
        ("纠结情况", indecision_score),
        ("疲劳水平", fatigue_score),  # 与脑疲劳一致
        ("恐惧感", fear_score),
        ("易怒", irritability_score),
        ("自闭", social_withdrawal_score),
    ]:
        lvl, lvl_name = level_from_score(sc)
        extra = None
        # 强制强调心理维度为代理
        extra = ""
        brief = standard_explanation(nm, lvl)
        ratings_psy.append(make_metric_record(nm, sc, lvl, lvl_name, brief))

    ratings_phys = []
    # 脑疲劳与心理疲劳一致（至少 level 一致）
    brain_fatigue_score = fatigue_score
    brain_fatigue_level, brain_fatigue_level_name = level_from_score(brain_fatigue_score)

    for nm, sc, note in [
        ("脑耗氧水平", oxygen_score, ""),
        ("脑疲劳水平", brain_fatigue_score, "与心理疲劳保持一致）"),
        ("血氧代谢水平", spo2_score, ""),
        ("大脑活力", vitality_score, None),
    ]:
        lvl, lvl_name = level_from_score(sc)
        brief = standard_explanation(nm, lvl)
        ratings_phys.append(make_metric_record(nm, sc, lvl, lvl_name, brief))

    ratings_cog = []
    for nm, sc, note in [
        ("脑稳定度", stability_score, None),
        ("抵抗干扰", anti_interf_score, None),
        ("思维弹性", flex_score, ""),
        ("反应速度", reaction_score, ""),
    ]:
        lvl, lvl_name = level_from_score(sc)
        brief = standard_explanation(nm, lvl)
        ratings_cog.append(make_metric_record(nm, sc, lvl, lvl_name, brief))

    ratings_sleep = []
    nm = "睡眠质量"
    lvl, lvl_name = level_from_score(sleep_score)
    brief = standard_explanation(nm, lvl)
    ratings_sleep.append(make_metric_record(nm, sleep_score, lvl, lvl_name, brief))

    # ========= 模块汇总 =========
    psy_sum = module_summary(ratings_psy)
    phys_sum = module_summary(ratings_phys)
    cog_sum = module_summary(ratings_cog)
    sleep_sum = module_summary(ratings_sleep)

    # ========= 输出成文本块 =========
    def _format_module(title, metrics, summ):
        lines = []
        ms = summ.get("module_score")
        ms_str = "NA" if ms is None else f"{ms:.1f}"
        lines.append(f"{title}（模块等级：{summ.get('module_level')}（{summ.get('module_level_name')}），模块风险分：{ms_str}）")
        lines.append(f"模块简述：{summ.get('module_brief')}")
        for m in metrics:
            lines.append(format_metric_line(m))
        return "\n".join(lines)

    rating_block = []
    rating_block.append("结构化评级（0-100，分数越高风险越高）：")
    rating_block.append(_format_module("1) 心理健康", ratings_psy, psy_sum))
    rating_block.append("")
    rating_block.append(_format_module("2) 脑生理监测", ratings_phys, phys_sum))
    rating_block.append("")
    rating_block.append(_format_module("3) 认知功能评估", ratings_cog, cog_sum))
    rating_block.append("")
    rating_block.append(_format_module("4) 睡眠情况", ratings_sleep, sleep_sum))
    rating_block.append("")
    rating_block.append("重要免责声明：以上评级为基于短窗EEG频谱/噪声/对称/调节等代理指标的规则映射，仅用于筛查与趋势参考，不构成临床诊断；请结合量表、访谈与专业检查综合判断。")

    rating_text = "\n".join(rating_block)

    # ========= 输出打包 =========
    result_dict = {
        "rel_alpha": rel_alpha,
        "rel_beta": rel_beta,
        "rel_theta": rel_theta,
        "theta_alpha_ratio": theta_alpha_ratio,
        "beta_alpha_ratio": beta_alpha_ratio,
        "emg_ratio": emg_ratio,
        "line_ratio": line_ratio,
        "spectral_entropy": spec_ent,
        "faa": faa,
        "symmetry_ok": symmetry_ok,
        "regulation_ok": regulation_ok,
        "fatigue_index": fatigue_index,

        # 新增：结构化评级与模块汇总（便于程序化使用）
        "ratings": {
            "心理健康": ratings_psy,
            "脑生理监测": ratings_phys,
            "认知功能评估": ratings_cog,
            "睡眠情况": ratings_sleep,
        },
        "module_summary": {
            "心理健康": psy_sum,
            "脑生理监测": phys_sum,
            "认知功能评估": cog_sum,
            "睡眠情况": sleep_sum,
        },
        "sleep_quality_text": sleep_quality_text,
    }

    # 保留原 result_text 并追加评级块
    result_text = (
        "功能检测结果（基于短窗EEG代理指标）：\n"
        + cog_text + "\n"
        + phys_text + "\n"
        + psy_text + "\n"
        + "\n"
        + rating_text
        + "\n"
    )
    return result_dict, result_text

# ========= 频段能量 + 对称性文字块 =========
def _load_edf_context(edf_file):
    try:
        return read_edf(edf_file)
    except ChannelMismatchError as e:
        info = {"错误": "通道不匹配", "原因": str(e)}
        if getattr(e, "details", None):
            info.update(e.details)
        brace_print_dict(info)
        sys.exit(1)
    except Exception as e:
        brace_print_dict({"错误": "未知异常", "exception": repr(e)})
        sys.exit(1)

def plot_topomaps_and_calculate_energy(edf_file=None, edf_ctx=None):
    if edf_ctx is None:
        if edf_file is None:
            raise ValueError("edf_file 和 edf_ctx 不能同时为空")
        eeg_data, _, sfreq, channel_positions, raw = _load_edf_context(edf_file)
    else:
        eeg_data, _, sfreq, channel_positions, raw = edf_ctx

    bands = [[0.5, 8], [8, 13], [14, 30]]
    band_names = ['慢波', 'α波', '快波']
    energy_stats = {}

    for i, band in enumerate(bands):
        filtered_data = mne.filter.filter_data(eeg_data, sfreq=sfreq, l_freq=band[0], h_freq=band[1], verbose=False)
        freqs, psd = compute_psd(filtered_data, sfreq)
        channel_energies = np.trapz(psd, freqs) * 1e12  # 转为 ~ μV^2 量级

        energy_stats[f'{band_names[i]}_max'] = float(np.max(channel_energies))
        energy_stats[f'{band_names[i]}_min'] = float(np.min(channel_energies))

        left_energies, right_energies = [], []
        for idx, (x, _) in channel_positions.items():
            if x < 0:
                left_energies.append(channel_energies[idx])
            elif x > 0:
                right_energies.append(channel_energies[idx])
        left_total = np.sum(left_energies) if len(left_energies) else np.nan
        right_total = np.sum(right_energies) if len(right_energies) else np.nan
        if np.isfinite(left_total) and np.isfinite(right_total) and right_total > 0:
            ratio = left_total / right_total
            symmetry_text = '左右基本对称' if (0.8 < ratio < 1.2) else '左右对称欠佳'
        else:
            symmetry_text = '对称性无法评估'
        energy_stats[f'{band_names[i]}_symmetry'] = symmetry_text

    report = f"""
脑电地形图（BEAM）特征：

α波能量：{energy_stats['α波_min']:.2f}–{energy_stats['α波_max']:.2f}（μV²近似），{energy_stats['α波_symmetry']}。
慢波能量：{energy_stats['慢波_min']:.2f}–{energy_stats['慢波_max']:.2f}（μV²近似），{energy_stats['慢波_symmetry']}。
快波能量：{energy_stats['快波_min']:.2f}–{energy_stats['快波_max']:.2f}（μV²近似），{energy_stats['快波_symmetry']}。
"""
    return report

# ========= 主频/描述 + 对称性/调节性写入报告 =========
def plot_spectrum_and_topomap(edf_file=None, edf_ctx=None):
    if edf_ctx is None:
        if edf_file is None:
            raise ValueError("edf_file 和 edf_ctx 不能同时为空")
        eeg_data, o1o2, sfreq, channel_positions, raw = _load_edf_context(edf_file)
    else:
        eeg_data, o1o2, sfreq, channel_positions, raw = edf_ctx

    # α–快波 4–30 Hz（用于总体主频描述）
    data_main = mne.filter.filter_data(eeg_data.copy(), sfreq=sfreq, l_freq=4, h_freq=30, verbose=False)
    psd_list = []
    for ch in range(data_main.shape[0]):
        freqs, psd = compute_psd(data_main[ch, :], sfreq)
        psd_list.append(psd)
    avg_psd = np.mean(np.array(psd_list), axis=0)
    main_low, main_high = compute_dominant_band_global_peak(freqs, avg_psd, band_range=(4, 30))
    rms_main_uv = float(np.sqrt(np.mean(np.square(data_main))) * 1e6)
    desc_main = amplitude_descriptor(rms_main_uv)

    # 慢波 4–8
    data_slow = mne.filter.filter_data(eeg_data.copy(), sfreq=sfreq, l_freq=4, h_freq=8, verbose=False)
    psd_list = []
    for ch in range(data_slow.shape[0]):
        freqs, psd = compute_psd(data_slow[ch, :], sfreq)
        psd_list.append(psd)
    avg_psd = np.mean(np.array(psd_list), axis=0)
    slow_low, slow_high = compute_dominant_band_global_peak(freqs, avg_psd, band_range=(4, 8))
    rms_slow_uv = float(np.sqrt(np.mean(np.square(data_slow))) * 1e6)
    desc_slow = amplitude_descriptor(rms_slow_uv)

    # 快波 14–30
    data_fast = mne.filter.filter_data(eeg_data.copy(), sfreq=sfreq, l_freq=14, h_freq=30, verbose=False)
    psd_list = []
    for ch in range(data_fast.shape[0]):
        freqs, psd = compute_psd(data_fast[ch, :], sfreq)
        psd_list.append(psd)
    avg_psd = np.mean(np.array(psd_list), axis=0)
    fast_low, fast_high = compute_dominant_band_global_peak(freqs, avg_psd, band_range=(14, 30))
    rms_fast_uv = float(np.sqrt(np.mean(np.square(data_fast))) * 1e6)
    desc_fast = amplitude_descriptor(rms_fast_uv)

    # === 成对导联对称性（30%标准）
    pair_results, symmetry_overall_text = check_symmetry_by_pairs(raw, eeg_data)

    # === 调节性（先给默认值，保证 reg_detail 一定存在）
    reg_detail = {"note": "缺少 O1/O2"}
    regulation_text = "调节无法评估（缺少 O1/O2）"

    if o1o2 is not None and o1o2.shape[0] in (1, 2):
        if o1o2.shape[0] == 2:
            o1, o2 = o1o2[0], o1o2[1]
            reg_detail, regulation_text = check_regulation_o1o2(o1, o2, fs=sfreq, segment_len_sec=3.0)
        else:
            reg_detail = {"note": "缺少 O2 或 O1"}
            regulation_text = "调节无法评估（缺少成对导联）"

    # === 功能检测三段结论 + 结构化评级（必须放在 reg_detail 得到之后）
    func_dict, func_text = compute_functional_results(
        raw=raw,
        window_data=eeg_data,
        sfreq=sfreq,
        pair_results=pair_results,
        reg_detail=reg_detail,
        line_freq=50
    )

    # 逐对附简要（可注释掉）
    pair_lines = []
    for item in pair_results:
        pair = item["pair"]
        if item["ratio"] is None:
            pair_lines.append(f"- {pair}: 无法评估（{item.get('note','')}）")
        else:
            a, b = item["amp_uv"]
            ratio = item["ratio"]
            ok = "对称" if item["symmetric"] else "不对称"
            pair_lines.append(f"- {pair}: RMS={a:.1f}/{b:.1f} μV，差值占比={ratio*100:.1f}% -> {ok}")
    pair_lines_text = "\n".join(pair_lines)

    report = f"""
脑电图（EEG）报告

主要波率：
各导联均以{desc_main}{(main_low or np.nan):.1f}-{(main_high or np.nan):.1f} Hz 波为主。

慢波：
各导联间有{desc_slow}{(slow_low or np.nan):.1f}-{(slow_high or np.nan):.1f} Hz 波。

快波：
各导联间有{desc_fast}{(fast_low or np.nan):.1f}-{(fast_high or np.nan):.1f} Hz 波。

波幅特点与对称性（成对导联 30% 标准）：
{symmetry_overall_text}
{pair_lines_text}

调节性：
{regulation_text}

认知/脑生理/心理健康 模块检测结果：
{func_text}
"""
    return report

# ========= 汇总生成 =========
def plot_topomaps_and_calculate_energy_block(edf_file=None, edf_ctx=None):
    return plot_topomaps_and_calculate_energy(edf_file=edf_file, edf_ctx=edf_ctx)

def Report_generation(edf_file, ascii_safe=None):
    """
    ascii_safe:
      - None: 自动判断（stdout 为 gbk/cp936 或环境变量 ASCII_SAFE=1 时启用）
      - True: 强制替换为安全字符
      - False: 保留原始字符（若 stdout 是 gbk 仍可能需要外层 try/except）
    """
    edf_ctx = _load_edf_context(edf_file)
    text1 = plot_spectrum_and_topomap(edf_ctx=edf_ctx)
    text2 = plot_topomaps_and_calculate_energy_block(edf_ctx=edf_ctx)
    text = text1 + "\n" + text2

    if ascii_safe is None:
        # 1) 环境变量优先（便于 Java 启动 exe 时控制）
        env_flag = (os.environ.get("ASCII_SAFE") or "").strip().lower()
        if env_flag in ("1", "true", "yes", "y", "on"):
            ascii_safe = True
        elif env_flag in ("0", "false", "no", "n", "off"):
            ascii_safe = False
        else:
            # 2) 自动：stdout 为 gbk/cp936 或未知（空）时启用
            out_enc = (getattr(sys.stdout, "encoding", "") or "").lower()
            ascii_safe = ("gbk" in out_enc) or ("cp936" in out_enc) or (out_enc == "")

    if ascii_safe:
        # 统一用安全化函数处理整段报告（包含 STANDARD_TEXT 的潜在危险符号）
        text = sanitize_for_console(text, target_encoding="gbk")

    return text


# ========= 最小自测（不依赖EDF，不影响主流程） =========
def run_self_test():
    print("=== SELF TEST: level boundaries ===")
    for s in [0, 65, 65.1, 85, 85.1, 95, 95.1, 100]:
        lvl, nm = level_from_score(s)
        print(f"score={s} -> {lvl}({nm})")

    print("\n=== SELF TEST: fatigue consistency ===")
    for fi in [1.0, 1.2, 1.6, 2.0, 2.4, 3.0]:
        fs = _score_fatigue(fi)
        lvl, nm = level_from_score(fs)
        # 心理疲劳与脑疲劳同源
        lvl2, nm2 = level_from_score(fs)
        ok = "OK" if lvl == lvl2 else "FAIL"
        print(f"fatigue_index={fi:.2f} -> fatigue_score={fs:.1f}, level={lvl}/{lvl2} [{ok}]")

# ========= 脚本入口 =========
if __name__ == "__main__":
    if "--self_test" in sys.argv:
        run_self_test()
        sys.exit(0)

    # 兼容：允许命令行传入 EDF 路径，否则使用默认示例
    edf_file = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Administrator\Desktop\临床bench\数据\包元宵25111310.edf"
    text = Report_generation(edf_file)
    print(text)
