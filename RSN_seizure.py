import os
import sys
import torch
import numpy as np
import mne
from scipy import signal

from model.RSN import RSN


# =========================
# 统一：通道不匹配异常 + 花括号打印（与原模型一致）
# =========================
class ChannelMismatchError(Exception):
    """用于在通道不匹配时携带详细原因。"""
    def __init__(self, msg, details=None):
        super().__init__(msg)
        self.details = details or {}


def brace_print_dict(d):
    """花括号打印（非 JSON），支持 dict/列表/字符串。"""
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


# =========================
# 读取 EDF + 构造 16 双极导联（与原模型一致风格：别名/歧义/缺失直接报错）
# =========================
def edf_eeg(file_path):
    """
    读取 EDF，校验/映射通道，若缺失则抛 ChannelMismatchError（含详细原因）。
    返回 shape=(16, T) 的双极导联数据（sfreq=200 重采样后）。
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False, encoding="latin1")
    except Exception as e:
        raise ChannelMismatchError("EDF文件读取失败", {"exception": repr(e), "file_path": file_path})

    sfreq = 200

    # 需要构建的 16 个双极导联（与原模型完全一致）
    bipolar_pairs = [
        ("Fp1", "F7"), ("F7", "T3"), ("T3", "T5"), ("T5", "O1"),
        ("Fp2", "F8"), ("F8", "T4"), ("T4", "T6"), ("T6", "O2"),
        ("Fp1", "F3"), ("F3", "C3"), ("C3", "P3"), ("P3", "O1"),
        ("Fp2", "F4"), ("F4", "C4"), ("C4", "P4"), ("P4", "O2"),
    ]

    # 别名表：兼容常见命名差异（与原模型一致）
    ALIASES = {
        "T3": ["T3", "T7"],
        "T4": ["T4", "T8"],
        "T5": ["T5", "P7"],
        "T6": ["T6", "P8"],
        "Fp1": ["Fp1", "FP1"],
        "Fp2": ["Fp2", "FP2"],
        "F3": ["F3"], "F4": ["F4"], "F7": ["F7"], "F8": ["F8"],
        "C3": ["C3"], "C4": ["C4"],
        "P3": ["P3"], "P4": ["P4"],
        "O1": ["O1"], "O2": ["O2"],
    }

    all_chs = raw.ch_names

    def find_best_match(keyword):
        """
        在原始通道中寻找包含 keyword 或其别名的通道（大小写不敏感）。
        优先匹配含 '-REF'/' REF' 的名称；如有多个候选，返回最优并记录歧义候选。
        返回: (matched_name, candidates_list)
        """
        candidates = []
        for alias in ALIASES.get(keyword, [keyword]):
            al = alias.lower()
            hits = [c for c in all_chs if al in c.lower()]
            candidates.extend(hits)

        # 去重（保序）
        candidates = list(dict.fromkeys(candidates))
        if not candidates:
            return None, []

        def score(name):
            n = name.lower()
            s = 0
            # 优先：-ref / ref
            if "-ref" in n or n.endswith(" ref") or n.endswith("- ref"):
                s += 2
            # 次优先：别名以更“完整”的形式出现
            for alias in ALIASES.get(keyword, [keyword]):
                if alias in name or alias.upper() in name:
                    s += 1
                    break
            return -s  # 升序排序用负数

        candidates_sorted = sorted(candidates, key=score)
        return candidates_sorted[0], candidates_sorted

    # 为每个关键位确定实际通道
    chosen_map = {}
    missing_keywords = []
    ambiguous = {}

    needed_keywords = sorted({k for a, b in bipolar_pairs for k in (a, b)})
    for kw in needed_keywords:
        match, cand = find_best_match(kw)
        if match is None:
            missing_keywords.append(kw)
        else:
            chosen_map[kw] = match
            if len(cand) > 1:
                ambiguous[kw] = cand[:6]

    if missing_keywords:
        raise ChannelMismatchError(
            "通道不匹配：缺少构建双极导联所需的原始通道",
            details={
                "缺失关键通道": missing_keywords,
                "文件包含通道数": len(all_chs),
                "文件示例通道": all_chs[:20],
                "提示": "支持大小写不敏感与常见别名(T3↔T7, T4↔T8, T5↔P7, T6↔P8)。请检查医院导出命名或改用别名。",
            }
        )

    unbuildable = []
    for a, b in bipolar_pairs:
        if a not in chosen_map or b not in chosen_map:
            unbuildable.append(f"{a}-{b}")
    if unbuildable:
        raise ChannelMismatchError(
            "通道不匹配：部分双极导联无法构造",
            details={
                "无法构造的导联": unbuildable,
                "已匹配映射": chosen_map,
                "歧义匹配": ambiguous if ambiguous else "无",
            }
        )

    # 选通道并重采样
    unique_chs = list(dict.fromkeys(chosen_map.values()))
    raw.pick_channels(unique_chs)
    raw.resample(sfreq=sfreq, npad="auto", n_jobs=1)

    data = raw.get_data()  # (n_ch, T)
    idx_map = {name: i for i, name in enumerate(raw.ch_names)}

    def chdata(keyword):
        return data[idx_map[chosen_map[keyword]]]

    # 构建 16 导双极数据
    channeled_data = np.zeros((16, data.shape[1]), dtype=np.float32)
    for i, (a, b) in enumerate(bipolar_pairs):
        channeled_data[i] = chdata(a) - chdata(b)

    return channeled_data


# =========================
# RSN 推理：输出/报错方式对齐原模型
# =========================
def supervised_rsn(file_path):
    # 1) EDF + 通道处理：按原模型风格报错
    try:
        X = edf_eeg(file_path)
    except ChannelMismatchError as e:
        info = {"错误": "通道不匹配", "原因": str(e)}
        if getattr(e, "details", None):
            info.update(e.details)
        brace_print_dict(info)
        sys.exit(1)
    except Exception as e:
        brace_print_dict({"错误": "未知异常", "exception": repr(e)})
        sys.exit(1)

    # 2) 输入预处理：保持你新模型原逻辑（2000点 -> (16,10,200)）
    try:
        X = signal.resample(X, 2000, axis=-1)     # (16, 2000)
        X = X.reshape(16, 10, 200)                # (16, 10, 200)
        X = torch.FloatTensor(X).unsqueeze(0)     # (1, 16, 10, 200)
    except Exception as e:
        brace_print_dict({"错误": "输入预处理失败", "exception": repr(e)})
        sys.exit(1)

    # 3) 模型加载 + 推理
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fine_tuning", "RSN.pth")
    if not os.path.exists(ckpt_path):
        brace_print_dict({"错误": "模型权重不存在", "ckpt_path": ckpt_path})
        sys.exit(1)

    try:
        model = RSN()
        checkpoint = torch.load(ckpt_path, map_location="cpu")

        # 兼容：有些pth会包一层 dict，如 {"state_dict": ...}
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        with torch.no_grad():
            pred = model(X)               # 期望 shape (1,1) 或 (1,)
            score_y = torch.sigmoid(pred)
            pred_y = torch.gt(score_y, 0.5).long()
    except Exception as e:
        brace_print_dict({"错误": "模型推理失败", "exception": repr(e), "ckpt_path": ckpt_path})
        sys.exit(1)

    # 4) 输出形式：对齐“trainer.test()[0] 打印 dict”的效果
    try:
        y_int = int(pred_y.reshape(-1)[0].item())
    except Exception:
        # 万一 pred_y 形状异常，也尽量给出可读输出
        y_int = int(pred_y.detach().cpu().numpy().reshape(-1)[0])

    out = {"检测结果": y_int}
    print(out)
    return out


if __name__ == "__main__":
    # 兼容：命令行传参优先，其次用你原来的默认路径
    if len(sys.argv) >= 2:
        file_path = sys.argv[1]
    else:
        file_path = r"C:\Users\jiang\Downloads\gpt2ref\gpt2ref\aaaaafoe_00000001.edf"

    supervised_rsn(file_path)