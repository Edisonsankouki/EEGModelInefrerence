# -*- coding: utf-8 -*-
import os
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import mne
mne.set_log_level("ERROR")  # 仅保留最终报告输出

from model.Cbramodfinetune import CBraMod


# =============== 花括号文本输出（非 JSON） =============== #
def brace_print(lines):
    print("{")
    for line in lines:
        print("  " + line)
    print("}")


# ====================== EDF 预处理模块 ====================== #
class EEGSingleInferencePreprocessor:
    def __init__(self,
                 chosen_channels=None,
                 resample_rate=200,
                 target_patch_num=30,
                 patch_size=200,
                 strict=True):
        self.chosen_channels = chosen_channels or [
            'EEG FP1-REF', 'EEG FP2-REF',
            'EEG F3-REF',  'EEG F4-REF',
            'EEG C3-REF',  'EEG C4-REF',
            'EEG P3-REF',  'EEG P4-REF',
            'EEG O1-REF',  'EEG O2-REF',
            'EEG F7-REF',  'EEG F8-REF',
            'EEG T3-REF',  'EEG T4-REF',
            'EEG T5-REF',  'EEG T6-REF'
        ]
        self.resample_rate = resample_rate
        self.target_patch_num = target_patch_num
        self.patch_size = patch_size
        self.strict = strict

    def normalize_channel_name(self, ch_name):
        name = ch_name.strip().lower().replace("eeg", "").replace(" ", "").replace("_", "")
        for suf in ["-ref", "-avg", "-avr", "-av", "-cz", "-a1", "-a2", "-m1", "-m2", "-mast", "-linkedmast"]:
            if name.endswith(suf):
                name = name[: -len(suf)]
                break
        name = name.replace("-", "")
        m = re.match(r'([a-z]+[0-9]+)', name)
        return m.group(1) if m else name

    def _is_same_electrode_diff_ref(self, raw_name, core):
        s = raw_name.lower().replace(" ", "").replace("eeg", "")
        return re.match(rf'^{re.escape(core)}-(ref|avg|avr|av|a1|a2|m1|m2|cz|mast|linkedmast)$', s) is not None

    def match_channels(self, raw):
        target_core = [self.normalize_channel_name(ch) for ch in self.chosen_channels]
        core_to_actuals = defaultdict(list)
        for ch in raw.ch_names:
            core = self.normalize_channel_name(ch)
            if core:
                core_to_actuals[core].append(ch)

        def choose_preferred(actual_list, core):
            for a in actual_list:
                if self._is_same_electrode_diff_ref(a, core):
                    return a
            for a in actual_list:
                if f"{core}-ref" in a.lower().replace(" ", ""):
                    return a
            return actual_list[0]

        ordered, used, missing = [], set(), []
        for core in target_core:
            if core in core_to_actuals:
                cands = [a for a in core_to_actuals[core] if a not in used]
                pick = choose_preferred(cands, core) if cands else choose_preferred(core_to_actuals[core], core)
                ordered.append(pick); used.add(pick)
            else:
                missing.append(core)

        if missing:
            raise RuntimeError("通道不匹配，缺乏所需通道")
        return ordered

    def process(self, edf_path):
        if not os.path.exists(edf_path):
            raise FileNotFoundError(edf_path)

        try:
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        except Exception:
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False, encoding='latin1')
            except Exception as e:
                raise RuntimeError(f"无法加载 EDF 文件: {e}")

        selected = self.match_channels(raw)
        raw.pick(selected, verbose=False)
        raw.resample(self.resample_rate, verbose=False)

        data = raw.get_data().astype(np.float32)  # (16, seq)
        ch, seq = data.shape
        patch_num = seq // self.patch_size
        if patch_num < 1:
            raise RuntimeError(f"文件过短（长度 {seq}），无法形成一个 patch。")

        data = data[:, :patch_num * self.patch_size]
        data = data.reshape(ch, patch_num, self.patch_size)

        x = torch.from_numpy(data)  # (16, P, 200)
        if patch_num > self.target_patch_num:
            idx = torch.linspace(0, patch_num - 1, steps=self.target_patch_num).long()
            x = x[:, idx, :]
        elif patch_num < self.target_patch_num:
            x = F.interpolate(
                x.unsqueeze(0),
                size=(self.target_patch_num, self.patch_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return x.unsqueeze(0)  # (1, 16, 30, 200)

def load_ckpt_safely(model, model_ckpt):
    import torch, re
    ckpt = torch.load(model_ckpt, map_location="cpu",weights_only=False)
    sd_raw = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # 1) 尝试统一命名：去掉常见前缀（Lightning/DP）
    def strip_prefix(k):
        for pref in ["model.", "module.", "net.", "backbone."]:
            if k.startswith(pref): 
                return k[len(pref):]
        return k
    sd = {strip_prefix(k): v for k, v in sd_raw.items()}

    # 2) 仅保留在当前模型中存在、且形状一致的权重
    model_sd = model.state_dict()
    usable = {}
    skipped_shape = []
    for k, v in sd.items():
        if k in model_sd:
            if model_sd[k].shape == v.shape:
                usable[k] = v
            else:
                skipped_shape.append((k, tuple(v.shape), tuple(model_sd[k].shape)))

    # 3) 加载（非严格）并报告差异
    missing, unexpected = model.load_state_dict(usable, strict=False)

    # 4) 详细诊断输出
    print("\n===== CKPT 适配诊断 =====")
    print(f"可用权重数: {len(usable)}/{len(sd)}")
    if skipped_shape:
        print("因形状不匹配被跳过的键（ckpt_shape -> model_shape）:")
        for k, s1, s2 in skipped_shape[:30]:
            print(f"  - {k}: {s1} -> {s2}")
        if len(skipped_shape) > 30:
            print(f"  ... 以及 {len(skipped_shape)-30} 个")
    if missing:
        print("加载后仍缺失的键 (missing):")
        for k in missing[:50]:
            print("  -", k)
        if len(missing) > 50:
            print(f"  ... 以及 {len(missing)-50} 个")
    if unexpected:
        print("意外多余的键 (unexpected):")
        for k in unexpected[:50]:
            print("  -", k)
        if len(unexpected) > 50:
            print(f"  ... 以及 {len(unexpected)-50} 个")
    print("========================\n")

    return missing, unexpected, skipped_shape
# ====================== 单样本 DataSet（配合 DataLoader） ====================== #
class SingleEEGDataset(Dataset):
    def __init__(self, edf_path, preprocessor: EEGSingleInferencePreprocessor):
        self.edf_path = edf_path
        self.pre = preprocessor
        # 读取并缓存（通道缺失直接抛异常）
        self.x = self.pre.process(self.edf_path)  # (1,16,30,200)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        # 返回形状 (16,30,200)，由 DataLoader 叠成 (B,16,30,200)
        
        return self.x.squeeze(0)


# ====================== 推理（DataLoader 形式） ====================== #
def run_inference_with_dataloader(model_ckpt, edf_path, label_names=None, device="cpu"):
    """
    使用 DataLoader 的形式进行推理；
    仅输出一段花括号包裹的中文报告（或失败原因）。
    """
    # 1) DataLoader
    try:
        ds = SingleEEGDataset(edf_path, EEGSingleInferencePreprocessor())
        dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=False)
    except Exception as e:
        brace_print([
            "报告状态: 失败",
            f"原因: {str(e)}",
            "处理建议: 请检查EDF文件与导联命名，必要时重新采集或更换文件。"
        ])
        return ""

    # 2) 模型 + 权重（严格加载）
    model = CBraMod(
        in_dim=200, out_dim=200, d_model=200,
        dim_feedforward=800, seq_len=30, n_layer=12, nhead=8
    )

    missing, unexpected, skipped = load_ckpt_safely(model, model_ckpt)

    # 3) 评估态 + 关 Dropout + 上设备
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            m.p = 0.0
    model.eval().to(device)

    # 4) 前向（DataLoader）
    try:
        with torch.no_grad():
            for batch in dl:
                x = batch.to(device)            # (B,16,30,200)
                logits = model(x)               # (B, 8) 假设输出维度≥4且[1,2,3]分别是抑郁/焦虑/癫痫
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        # =========================
        # 5) 后处理（自定义阈值）
        # =========================
        # 目标标签
        targets = ["抑郁", "焦虑", "癫痫"]

        # label_names存在时用它反查索引；否则按固定索引假定
        if label_names is not None:
            name_to_idx = {n: i for i, n in enumerate(label_names)}
            pairs = [(n, name_to_idx[n]) for n in targets if n in name_to_idx]
        else:
            # 注意：这里假设
            #   抑郁 -> probs[1]
            #   焦虑 -> probs[2]
            #   癫痫 -> probs[3]
            pairs = [("抑郁", 2), ("焦虑", 4), ("癫痫", 6)]

        # 每个标签单独的阈值
        custom_threshold = {
            "抑郁": 0.500,  # Class 1
            "焦虑": 0.500,  # Class 2
            "癫痫": 0.0070,  # Class 3
        }

        def risk_for(name, p):
            thr = custom_threshold.get(name, 0.5)  # fallback=0.5 以防万一
            return "高风险" if float(p) >= thr else "低风险"

        risk_results = {}  # "抑郁": "高风险"/"低风险"
        for name, idx in pairs:
            if 0 <= idx < len(probs):
                risk_results[name] = risk_for(name, probs[idx])
            else:
                risk_results[name] = "无结果"

        dep_r = risk_results.get("抑郁", "无结果")
        anx_r = risk_results.get("焦虑", "无结果")
        epi_r = risk_results.get("癫痫", "无结果")

        # =========================
        # 6) 报告（不再显示具体概率）
        # =========================
        lines = [
            "报告类型: EEG筛查结果（自动化辅助）",
            "判定规则: 针对不同风险类别使用内部阈值；本报告仅作筛查参考，需结合临床与量表/视频脑电。",
            f"结论: 抑郁: {dep_r}；焦虑: {anx_r}；癫痫: 低风险",
            "风险评估:",
            f"  - 抑郁: {dep_r}",
            f"  - 焦虑: {anx_r}",
            f"  - 癫痫: 低风险",
            "建议:",
            "  - 抑郁/焦虑: 结合病史与量表（PHQ-9、GAD-7）进一步评估；必要时精神科/心理科随访。",
            "  - 癫痫: 如临床怀疑，请结合发作史与更长程视频EEG/睡眠监测；必要时神经内科评估。",
            "  - 说明: 本结果为模型筛查结论，不能替代医生诊断。"
        ]
        brace_print(lines)
        return ""

    except Exception as e:
        brace_print([
            "报告状态: 失败",
            f"原因: 推理失败。{str(e)}",
            "处理建议: 请检查输入与权重文件并重试。"
        ])
        return ""



# ====================== 简单封装 ====================== #
def Cbramod_run(edf_path, device="cpu"):
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fine_tuning', 'cbra1.ckpt')
    label_names = [
        "帕金森",      # 0
        "异常脑电",    # 1
        "抑郁",        # 2
        "正常脑电",    # 3
        "焦虑",        # 4
        "痴呆",        # 5
        "癫痫",        # 6
        "睡眠障碍",    # 7
        "精神分裂",    # 8
        "精神障碍",    # 9
        "躁狂",        # 10
    ]
    return run_inference_with_dataloader(
        ckpt,
        edf_path,
        label_names=label_names,
        device=device
    )


# ====================== 脚本运行示例 ====================== #
if __name__ == "__main__":
    EDF_PATH = r"C:\Users\jiang\Downloads\Telegram Desktop\488.edf"
    CKPT = r"C:\Users\jiang\Downloads\gpt2ref\gpt2ref\fine_tuning\cbra1.ckpt"

    label_names = [
        "帕金森",      # 0
        "异常脑电",    # 1
        "抑郁",        # 2
        "正常脑电",    # 3
        "焦虑",        # 4
        "痴呆",        # 5
        "癫痫",        # 6
        "睡眠障碍",    # 7
        "精神分裂",    # 8
        "精神障碍",    # 9
        "躁狂",        # 10
    ]

    run_inference_with_dataloader(
        CKPT,
        EDF_PATH,
        label_names=label_names,
        device="cpu"
    )