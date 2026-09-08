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

def edf_eeg(file_path):
    # 读取原始数据并获取采样率
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False, encoding='latin1')
    sfreq = 200

    # 定义需要提取的通道名称的关键字
    keywords = ['Fp1', 'F7', 'Fp2', 'F8', 'O2', 'C3', 'F3', 'P3', 'P4', 'C4', 'T3', 'T5', 'T4', 'T6', 'O1','F4']

    # 获取EDF文件中的所有通道名称
    all_channel_names = raw.ch_names
    
    # 找到包含关键字的通道
    selected_channels = [ch for ch in all_channel_names if any(keyword.lower() in ch.lower() for keyword in keywords)]
    
    # 提取指定的通道
    raw.pick_channels(selected_channels)
    
    # 重采样数据
    raw.resample(sfreq=sfreq, npad='auto', n_jobs=1)
    
    # 获取通道名称和数据
    raw_data = raw.get_data()

    # 创建映射字典以便于索引
    ch_name_to_index = {name: idx for idx, name in enumerate(raw.ch_names)}
    
    # 动态获取索引
    def get_channel_index(keyword):
        for ch_name in ch_name_to_index:
            if keyword in ch_name:
                return ch_name_to_index[ch_name]
        print(f"Channel with keyword '{keyword}' not found.")
        raise ValueError(f"Channel with keyword '{keyword}' not found.")

    channeled_data = raw_data.copy()[:16]
    channeled_data[0] = raw_data[get_channel_index("Fp1")]- raw_data[get_channel_index("F7")]
    channeled_data[1] = raw_data[get_channel_index("F7")]- raw_data[get_channel_index("T3")]
    channeled_data[2] = raw_data[get_channel_index("T3")]- raw_data[get_channel_index("T5")]
    channeled_data[3] = raw_data[get_channel_index("T5")]- raw_data[get_channel_index("O1")]
    channeled_data[4] = raw_data[get_channel_index("Fp2")]- raw_data[get_channel_index("F8")]
    channeled_data[5] = raw_data[get_channel_index("F8")]- raw_data[get_channel_index("T4")]
    channeled_data[6] = raw_data[get_channel_index("T4")]- raw_data[get_channel_index("T6")]
    channeled_data[7] = raw_data[get_channel_index("T6")]- raw_data[get_channel_index("O2")]
    channeled_data[8] = raw_data[get_channel_index("Fp1")]- raw_data[get_channel_index("F3")]
    channeled_data[9] = raw_data[get_channel_index("F3")]- raw_data[get_channel_index("C3")]
    channeled_data[10] = raw_data[get_channel_index("C3")]- raw_data[get_channel_index("P3")]
    channeled_data[11] = raw_data[get_channel_index("P3")]- raw_data[get_channel_index("O1")]
    channeled_data[12] = raw_data[get_channel_index("Fp2")]- raw_data[get_channel_index("F4")]
    channeled_data[13] = raw_data[get_channel_index("F4")]- raw_data[get_channel_index("C4")]
    channeled_data[14] = raw_data[get_channel_index("C4")]- raw_data[get_channel_index("P4")]
    channeled_data[15] = raw_data[get_channel_index("P4")]- raw_data[get_channel_index("O2")]
    
    return channeled_data

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
        result = np.array([])

        for out in test_step_outputs:
            result = np.append(result, out[0])

        if len(result)!=0:
            y_pred = result.copy()
            y_pred[y_pred >= self.threshold] = 1
            y_pred[y_pred < self.threshold] = 0
            self.log("检测结果", y_pred.copy().item(), sync_dist=True)
            return y_pred

def get_ad_text():
    """返回AD患者（认知功能障碍）的固定报告文本"""
    return """
    【认知功能障碍评估报告】
    
    基于脑电信号（EEG）分析的检测结果显示，受检者存在认知功能障碍的相关脑电特征。
    
    认知功能障碍（Cognitive Impairment）是一种常见的神经系统退行性表现，其早期症状可能包括记忆力减退、注意力下降、执行功能受损以及语言能力变化等。脑电信号分析能够反映大脑皮层神经活动的异常模式，为认知功能评估提供重要的客观指标。
    
    本检测结果提示受检者的脑电节律（如α波、θ波等）出现异常改变，这可能与神经元突触功能减退、神经递质失衡或脑网络连接异常有关。建议受检者及时前往神经内科或记忆门诊进行进一步的专业评估，包括神经心理学量表测试、影像学检查等，以便明确诊断并制定个性化的干预方案。
    
    早期识别和干预对于延缓认知功能下降、改善生活质量具有重要意义。建议定期进行认知功能评估，关注日常生活能力的细微变化，同时保持积极的认知训练和社交活动。
    
    【温馨提示】本结果为初步筛查指标，不能替代专业医学诊断，请咨询专科医生获取权威诊疗建议。
    """

def get_normal_text():
    """返回正常检测结果的固定文本"""
    return """
    【认知功能评估报告】
    
    基于脑电信号（EEG）分析的检测结果显示，受检者的脑电活动未见明显异常特征，符合正常认知功能表现的脑电模式。
    
    检测结果表明受检者的脑电节律（包括α波、β波、θ波等频段）处于正常范围，大脑皮层神经元活动呈现良好的同步性和协调性，未发现典型的认知功能障碍相关脑电标志物。
    
    建议继续保持健康的生活方式，包括规律作息、均衡饮食、适度运动和积极的社会交往，这些都有助于维持良好的认知功能状态。同时，建议定期进行健康体检，关注自身认知功能的变化。
    
    【温馨提示】本结果为初步筛查指标，建议结合其他临床检查综合评估认知功能状况。
    """

def supervised(file_path):
    try:
        X = edf_eeg(file_path)
    except Exception as e:
        print("通道不匹配，缺乏所需通道")
        sys.exit(1)

    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fine_tuning', 'seizure.ckpt')

    mean = np.mean(X, axis=-1, keepdims=True)
    std = np.std(X, axis=-1, keepdims=True)
    std = np.maximum(std, 1e-8)
    X = (X - mean) / std
    X = torch.FloatTensor(X)
    X = X.unsqueeze(0)  
    X = X.unsqueeze(0)  

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

    # 获取预测结果
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path=ckpt_path, dataloaders=X
    )[0]
    
    print(pretrain_result)
    
    # 解析预测结果并输出对应的文本
    # 假设pretrain_result中包含了检测结果，根据您的代码，结果应该存储在"检测结果"键中
    if "检测结果" in pretrain_result:
        prediction = pretrain_result["检测结果"]
    else:
        # 如果直接返回的是数值，取第一个值
        prediction = list(pretrain_result.values())[0] if pretrain_result else 0
    
    # 确保prediction是整数
    prediction = int(prediction) if isinstance(prediction, (int, float, np.number)) else 0
    
    if prediction == 1:
        print("\n" + "="*80)
        print("检测结果：疑似认知功能障碍（AD倾向）")
        print("="*80)
        print(get_ad_text())
        print("="*80)
    else:
        print("\n" + "="*80)
        print("检测结果：认知功能正常")
        print("="*80)
        print(get_normal_text())
        print("="*80)
    
    return pretrain_result

if __name__ == "__main__":
    # 替换为.edf文件路径  
    file_path = "20370.edf"
    supervised(file_path)