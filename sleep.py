import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies import DDPStrategy
import os
import re
from model import (
    gpt4ts,
    # labram_base_patch200_200,
)
import mne
import pandas as pd
import matplotlib.pyplot as plt
import csv
from pytorch_lightning.strategies import SingleDeviceStrategy
import sys
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

# 划分数据为30秒不重叠的样本
# ===== 重写：将 EDF 按 30s 切片，并对通道做“别名-优先级-歧义”匹配与报错 =====
def divide_into_samples(file_path, sample_length=30):
    """
    返回 shape = (num_segments, num_channels, seg_points)
    目标通道（按优先级）: Fpz, Cz, Pz, Oz
    至少需要命中 2 个通道；若 <2，则抛出 ChannelMismatchError 并给出详细原因。
    """
    try:
        raw_data = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    except Exception as e:
        raise ChannelMismatchError("EDF 文件读取失败", {"exception": repr(e)})

    sfreq = float(raw_data.info.get("sfreq", 0.0))
    if sfreq <= 0:
        raise ChannelMismatchError("采样率读取异常", {"sfreq": raw_data.info.get("sfreq")})

    # 目标关键通道（大小写不敏感）及常见别名
    TARGETS = ["Fpz", "Cz", "Pz", "Oz"]
    ALIASES = {
        "Fpz": ["Fpz", "FPz"],
        "Cz":  ["Cz", "CZ"],
        "Pz":  ["Pz", "PZ"],
        "Oz":  ["Oz", "OZ"],
    }

    all_chs = raw_data.ch_names
    all_lower = [c.lower() for c in all_chs]

    def candidates_for(keyword):
        cands = []
        for alias in ALIASES.get(keyword, [keyword]):
            al = alias.lower()
            for name in all_chs:
                if al in name.lower():
                    cands.append(name)
        # 去重
        cands = list(dict.fromkeys(cands))
        return cands

    def score(name, keyword):
        n = name.lower()
        s = 0
        # 优先含有 "-REF" 或以 " ref" 结尾（常见参考电极命名）
        if "-ref" in n or n.endswith(" ref") or n.endswith("- ref"):
            s += 3
        # 包含精确大小写别名，稍加分
        if any(alias in name for alias in ALIASES.get(keyword, [keyword])):
            s += 1
        # 词首/词尾更优（简单启发）
        base = keyword.lower()
        if n.startswith(base) or n.endswith(base):
            s += 1
        return -s  # sort ascending

    chosen = {}          # {"Fpz": "EEG FPz-REF", ...}
    ambiguous = {}       # {"Fpz": [候选...], ...}
    missing = []         # ["Fpz", ...]

    for kw in TARGETS:
        cand = candidates_for(kw)
        if not cand:
            missing.append(kw)
            continue
        cand_sorted = sorted(cand, key=lambda nm: score(nm, kw))
        chosen[kw] = cand_sorted[0]
        if len(cand_sorted) > 1:
            ambiguous[kw] = cand_sorted[:8]

    # 需要至少 2 个通道
    matched_count = len(chosen)
    if matched_count < 2:
        raise ChannelMismatchError(
            "通道不匹配：可用 EEG 通道少于 2 个，无法进行稳定分期",
            {
                "缺失关键通道": missing if missing else "无（但整体命中不足 2 个）",
                "已匹配映射": chosen if chosen else "无",
                "歧义匹配": ambiguous if ambiguous else "无",
                "文件包含通道数": len(all_chs),
                "文件示例通道": all_chs[:20],
                "提示": "建议保证至少 Fpz/Cz/Pz/Oz 中的 2-4 个；命名大小写不敏感，优先选择包含 -REF 的通道。",
            }
        )

    # 选出最多 4 个（按 TARGETS 顺序保序），并进行 pick/重采样
    ordered = [kw for kw in TARGETS if kw in chosen]
    pick_names = [chosen[kw] for kw in ordered]
    # 去重（避免不同 kw 命中到同一原始通道）
    pick_names = list(dict.fromkeys(pick_names))

    try:
        raw_data.pick_channels(pick_names)
    except Exception as e:
        raise ChannelMismatchError("选通道失败（pick_channels）", {
            "待选通道": pick_names,
            "exception": repr(e),
        })

    try:
        raw_data.resample(sfreq=sfreq, npad="auto", n_jobs=1)
    except Exception as e:
        raise ChannelMismatchError("重采样失败（resample）", {"sfreq": sfreq, "exception": repr(e)})

    seg_points = int(sample_length * sfreq)
    if seg_points <= 0:
        raise ChannelMismatchError("分段长度非法", {"sample_length": sample_length, "sfreq": sfreq})

    total_pts = int(raw_data.n_times)
    n_segments = total_pts // seg_points
    if n_segments == 0:
        raise ChannelMismatchError("有效数据不足以切 1 段", {
            "总点数": total_pts,
            "每段点数": seg_points,
            "采样率": sfreq,
            "文件包含通道": raw_data.ch_names,
        })

    data = raw_data.get_data()  # shape: (num_channels, total_pts)
    samples = []
    for i in range(n_segments):
        start = i * seg_points
        end = (i + 1) * seg_points
        samples.append(data[:, start:end])

    samples = np.asarray(samples)  # (n_segments, n_ch, seg_points)
    return samples


class LitModel_finetune(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model


    def test_step(self, batch, batch_idx):
        X = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = convScore.cpu().numpy()
            # print("convScore 的形状:", convScore.shape)

        return step_result

    def test_epoch_end(self, test_step_outputs):
        result = []
        gt = np.array([])
        for out in test_step_outputs:
            result.append(np.argmax(out[0], axis=0))
        # result = np.concatenate(result, axis=0)
        # result = multiclass_metrics_fn(
        #     gt, result, metrics=["accuracy","balanced_accuracy", "cohen_kappa", "f1_weighted"]
        # )
        result = np.array(result)

        # result_str = str(result)
        # 打开CSV文件用于追加写入 a
        with open('classifications.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for value in result:
                row = [value]
                writer.writerow(row)
        # result = np.argmax(result, axis=0)
        # self.log("result", result, sync_dist=True)
        return result



def supervised(X,ckpt_path):
    mean = np.mean(X, axis=-1, keepdims=True)
    std = np.std(X, axis=-1, keepdims=True)
    std = np.maximum(std, 1e-8)
    X = (X - mean) / std
    X = torch.FloatTensor(X)
    X = X.unsqueeze(1)  
    # X = X.unsqueeze(0)  
    class_number = 5

    model = gpt4ts(class_number)
    lightning_model = LitModel_finetune(model)


    trainer = pl.Trainer(
        devices="auto",
        accelerator="cpu",
        strategy=SingleDeviceStrategy(device="cpu"),
        auto_select_gpus=True,
        benchmark=True,
        enable_checkpointing=True,
    )


    # test the model
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path=ckpt_path, dataloaders= X
    )[0]
    return pretrain_result



# ===== 重写：sleep() 里捕获并打印详细原因 =====
def sleep(file_path):
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fine_tuning', 'gpt4ts-sleep.ckpt')
    batch_size = 512

    # 清理历史 CSV（避免不同文件追加混淆）
    if os.path.exists('classifications.csv'):
        os.remove('classifications.csv')

    try:
        samples = divide_into_samples(file_path, sample_length=30)
    except ChannelMismatchError as e:
        info = {"错误": "通道不匹配", "原因": str(e)}
        if getattr(e, "details", None):
            info.update(e.details)
        brace_print_dict(info)
        sys.exit(1)
    except Exception as e:
        brace_print_dict({"错误": "未知异常", "exception": repr(e)})
        sys.exit(1)

    # 分批推理
    num_batches = len(samples) // batch_size + (1 if len(samples) % batch_size != 0 else 0)
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(samples))
        batch_samples = samples[start:end]
        _ = supervised(batch_samples, ckpt_path)

    # 分期映射
    sleep_stage_mapping = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}

    # 读取结果
    with open('classifications.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        result = [int(row[0]) for row in reader]

    # 平滑（多数投票窗口）
    def smooth_classifications(result, window_size=20):
        smoothed = []
        for i in range(len(result)):
            window = result[max(0, i - window_size // 2):min(len(result), i + window_size // 2 + 1)]
            smoothed.append(max(set(window), key=window.count))
        return smoothed

    smoothed_result = smooth_classifications(result)
    total_duration = len(smoothed_result) * 30  # 秒
    current_stage = smoothed_result[0]
    stage_duration = 0
    output_text = ""

    output_text += f"总时长: {total_duration / 60:.2f} 分钟<br>"
    for i in range(len(smoothed_result)):
        if smoothed_result[i] == current_stage:
            stage_duration += 30
        else:
            output_text += f"睡眠分期: {sleep_stage_mapping[current_stage]}, 时长: {stage_duration / 60} 分钟<br>"
            current_stage = smoothed_result[i]
            stage_duration = 30
    if stage_duration > 0:
        output_text += f"睡眠分期: {sleep_stage_mapping[current_stage]}, 时长: {stage_duration / 60} 分钟<br>"
        output_text += "睡眠分期检测结束！<br>"

    os.remove('classifications.csv')
    return output_text
if __name__ == "__main__":


    file_path = "SC4102E0-PSG.edf"
    print(os.path.join(os.path.dirname(os.path.abspath(__file__)),  'fine_tuning', 'gpt4ts-sleep.ckpt'))

    sleep(file_path)
