import torch
import numpy as np
from scipy.signal import resample  
import pickle
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import os
import re
from pytorch_lightning.strategies import SingleDeviceStrategy
from model import (
    gpt4ts,
    labram_base_patch200_200,
)
import mne
import pandas as pd
import sys


class LitModel_finetune(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.threshold = 0.4153

    def test_step(self, batch, batch_idx):
        X = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = torch.sigmoid(convScore).cpu().numpy()
        return step_result

    def test_epoch_end(self, test_step_outputs):
        import numpy as np
        result = np.array([])

        for out in test_step_outputs:
            result = np.append(result, out[0])

        if len(result) != 0:
            y_pred = result.copy()
            y_pred[y_pred >= self.threshold] = 1
            y_pred[y_pred <  self.threshold] = 0
            self.log("检测结果", int(y_pred.copy()), sync_dist=True)
            return y_pred
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
                print(f"    {k2}: {v2}")
        else:
            print(f"  {k}: {v}")
    print("}")

def edf_eeg(file_path):
    """
    读取 EDF，校验/映射通道，若缺失则抛出 ChannelMismatchError，错误中包含详细原因。
    返回 shape=(16, T) 的双极导联数据。
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False, encoding='latin1')
    except Exception as e:
        raise ChannelMismatchError("EDF文件读取失败", {"exception": repr(e)})

    sfreq = 200

    # 需要构建的 16 个双极导联（以“关键字”表示，大小写不敏感，支持别名）
    bipolar_pairs = [
        ("Fp1","F7"), ("F7","T3"), ("T3","T5"), ("T5","O1"),
        ("Fp2","F8"), ("F8","T4"), ("T4","T6"), ("T6","O2"),
        ("Fp1","F3"), ("F3","C3"), ("C3","P3"), ("P3","O1"),
        ("Fp2","F4"), ("F4","C4"), ("C4","P4"), ("P4","O2"),
    ]

    # 别名表（常见命名差异）：T3/T4/T5/T6 <-> T7/T8/P7/P8，O1/O2 常写作 O1-REF 等
    ALIASES = {
        "T3": ["T3", "T7"],
        "T4": ["T4", "T8"],
        "T5": ["T5", "P7"],
        "T6": ["T6", "P8"],
        "Fp1": ["Fp1", "FP1"],
        "Fp2": ["Fp2", "FP2"],
        "F3": ["F3"], "F4": ["F4"], "F7": ["F7"], "F8":["F8"],
        "C3": ["C3"], "C4": ["C4"],
        "P3": ["P3"], "P4": ["P4"],
        "O1": ["O1"], "O2": ["O2"],
    }

    # 原始通道名
    all_chs = raw.ch_names
    all_chs_lower = [c.lower() for c in all_chs]

    def find_best_match(keyword):
        """
        在原始通道中寻找包含某 keyword 或其别名的通道。
        优先匹配包含 '-REF' 或 ' -REF' 的名称；如有多个候选，返回第一个并记录歧义。
        返回: (matched_name, candidates_list)
        若找不到，返回 (None, [])
        """
        candidates = []
        for alias in ALIASES.get(keyword, [keyword]):
            al = alias.lower()
            hits = [c for c in all_chs if al in c.lower()]
            candidates.extend(hits)

        # 去重
        candidates = list(dict.fromkeys(candidates))

        if not candidates:
            return None, []

        # 优先含有 -REF / REF 结尾的名称
        def score(name):
            n = name.lower()
            s = 0
            if "-ref" in n or n.endswith(" ref") or n.endswith("- ref"):
                s += 2
            # 次优先：完全词边界命中
            # 这里只做简单启发：如果 alias 大写形式作为独立子串出现，加 1 分
            for alias in ALIASES.get(keyword, [keyword]):
                if alias in name or alias.upper() in name:
                    s += 1
                    break
            return -s  # sort ascending, so use negative

        candidates_sorted = sorted(candidates, key=score)
        return candidates_sorted[0], candidates_sorted

    # 为每个原始“关键位”确定一个实际通道
    chosen_map = {}          # 例如 {"Fp1": "EEG FP1-REF", ...}
    missing_keywords = []    # 完全找不到的关键字
    ambiguous = {}           # 歧义匹配: {"F7": ["EEG F7-REF","F7-A1",...]}

    for kw in sorted({k for pair in bipolar_pairs for k in pair}):
        match, cand = find_best_match(kw)
        if match is None:
            missing_keywords.append(kw)
        else:
            chosen_map[kw] = match
            if len(cand) > 1:
                ambiguous[kw] = cand[:6]  # 最多展示前6个候选，避免过长

    # 如果有缺失，直接报错并给出细节
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

    # 进一步检查：哪几个双极对无法构造（因为某一侧虽然找到，但另一侧找不到或重复名）
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
    unique_chs = list(set(chosen_map.values()))
    raw.pick_channels(unique_chs)
    raw.resample(sfreq=sfreq, npad='auto', n_jobs=1)

    data = raw.get_data()
    idx_map = {name: i for i, name in enumerate(raw.ch_names)}

    def chdata(keyword):
        return data[idx_map[chosen_map[keyword]]]

    # 构建 16 导双极数据
    channeled_data = data.copy()[:16]
    pairs = bipolar_pairs  # 简写
    for i, (a, b) in enumerate(pairs):
        channeled_data[i] = chdata(a) - chdata(b)

    return channeled_data

def supervised(file_path):
    try:
        X = edf_eeg(file_path)
    except ChannelMismatchError as e:
        # 结构化打印详细原因
        info = {"错误": "通道不匹配", "原因": str(e)}
        if getattr(e, "details", None):
            info.update(e.details)
        brace_print_dict(info)
        sys.exit(1)
    except Exception as e:
        brace_print_dict({"错误": "未知异常", "exception": repr(e)})
        sys.exit(1)

    # ====== 下面保持你的原逻辑不变 ======
    import numpy as np
    import torch
    from pytorch_lightning.strategies import SingleDeviceStrategy
    from model import gpt4ts

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fine_tuning', 'seizure.ckpt')

    mean = np.mean(X, axis=-1, keepdims=True)
    std = np.std(X, axis=-1, keepdims=True)
    std = np.maximum(std, 1e-8)
    X = (X - mean) / std
    X = torch.FloatTensor(X).unsqueeze(0).unsqueeze(0)

    model = gpt4ts(1)
    lightning_model = LitModel_finetune(model)

    trainer = pl.Trainer(
        devices='auto',
        accelerator="cpu",
        strategy=SingleDeviceStrategy(device="cpu"),
        auto_select_gpus=True,
        benchmark=True,
        enable_checkpointing=True,
    )

    pretrain_result = trainer.test(model=lightning_model, ckpt_path=ckpt_path, dataloaders=X)[0]
    print(pretrain_result)
    return pretrain_result