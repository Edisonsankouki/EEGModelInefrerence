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
from scipy.signal import resample
from pytorch_lightning.strategies import SingleDeviceStrategy
import sys


def edf_eeg(file_path):
    # 读取原始数据并获取采样率
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False,encoding='latin1')
    sfreq = 200

    # 定义需要提取的通道名称的关键字
    keywords = ['Fp1', 'F7', 'Fp2', 'F8', 'O2', 'C3', 'F3', 'P3', 'P4', 'C4', 'T3', 'T5', 'T4', 'T6', 'O1','F4']
    # 获取EDF文件中的所有通道名称
    all_channel_names = raw.ch_names
    
    # 找到包含关键字的通道
    selected_channels = [ch for ch in all_channel_names if any(keyword in ch for keyword in keywords)]
    
    # 提取指定的通道
    raw.pick_channels(selected_channels)
    
    # 重采样数据
    raw.resample(sfreq=sfreq, npad='auto', n_jobs=1)
    
    # 获取通道名称和数据
    raw_data = raw.get_data()

    # 创建映射字典以便于索引
    ch_name_to_index = {name: idx for idx, name in enumerate(raw.ch_names)}

    # 创建新的通道数据
    channeled_data = raw_data[:10].copy()
    
    # 动态获取索引
    def get_channel_index(keyword):
        for ch_name in ch_name_to_index:
            if keyword in ch_name:
                return ch_name_to_index[ch_name]
        raise ValueError(f"Channel with keyword '{keyword}' not found.")


    # channeled_data = raw_data.copy()[:16]
    # channeled_data[0] = raw_data[get_channel_index("Fp1")]- raw_data[get_channel_index("F7")]
    # channeled_data[1] = raw_data[get_channel_index("F7")]- raw_data[get_channel_index("T3")]
    # channeled_data[2] = raw_data[get_channel_index("T3")]- raw_data[get_channel_index("T5")]
    # channeled_data[3] = raw_data[get_channel_index("T5")]- raw_data[get_channel_index("O1")]
    # channeled_data[4] = raw_data[get_channel_index("Fp2")]- raw_data[get_channel_index("F8")]
    # channeled_data[5] = raw_data[get_channel_index("F8")]- raw_data[get_channel_index("T4")]
    # channeled_data[6] = raw_data[get_channel_index("T4")]- raw_data[get_channel_index("T6")]
    # channeled_data[7] = raw_data[get_channel_index("T6")]- raw_data[get_channel_index("O2")]
    # channeled_data[8] = raw_data[get_channel_index("Fp1")]- raw_data[get_channel_index("F3")]
    # channeled_data[9] = raw_data[get_channel_index("F3")]- raw_data[get_channel_index("C3")]
    # channeled_data[10] = raw_data[get_channel_index("C3")]- raw_data[get_channel_index("P3")]
    # channeled_data[11] = raw_data[get_channel_index("P3")]- raw_data[get_channel_index("O1")]
    # channeled_data[12] = raw_data[get_channel_index("Fp2")]- raw_data[get_channel_index("F4")]
    # channeled_data[13] = raw_data[get_channel_index("F4")]- raw_data[get_channel_index("C4")]
    # channeled_data[14] = raw_data[get_channel_index("C4")]- raw_data[get_channel_index("P4")]
    # channeled_data[15] = raw_data[get_channel_index("P4")]- raw_data[get_channel_index("O2")]
    
    return channeled_data

# 划分数据为30秒不重叠的样本
def divide_into_samples(file_path, sample_length=5):
    # 计算每个样本的采样点数
    raw_data = mne.io.read_raw_edf(file_path, preload=True,encoding='latin1')

    sfreq = raw_data.info['sfreq']
    # sfreq =200
    sample_points = int(sample_length * sfreq)
    

    keywords = ['FP1', 'F7', 'FP2', 'F8', 'O2', 'C3', 'F3', 'P3', 'P4', 'C4']

    # 获取EDF文件中的所有通道名称
    all_channel_names = raw_data.ch_names
    

    selected_channels = [ch for ch in all_channel_names if any(keyword.lower() in ch.lower() for keyword in keywords)]
    if not selected_channels:  # 如果列表为空
        raise ValueError(
            f"No channels matched the keywords: {keywords}. "
            f"Available channels: {all_channel_names}"
        )
    
    
    # 提取指定的通道
    raw_data.pick_channels(selected_channels)
    # raw_data.resample(sfreq=sfreq, npad='auto', n_jobs=1)
    n_samples = int(raw_data.n_times / sample_points)

    samples = []
    for i in range(n_samples):
        start = i * sample_points
        end = (i + 1) * sample_points
        sample = raw_data.get_data(start=start, stop=end)
        samples.append(sample) 
    return np.array(samples)


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
        with open('classifications_spike.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for value in result:
                row = [value]
                writer.writerow(row)
        # result = np.argmax(result, axis=0)
        # self.log("result", result, sync_dist=True)
        return result



def supervised(X,ckpt_path):
    X = resample(X,200*5, axis=-1)
    mean = np.mean(X, axis=-1, keepdims=True)
    std = np.std(X, axis=-1, keepdims=True)
    std = np.maximum(std, 1e-8)
    X = (X - mean) / std
    
    X = torch.FloatTensor(X)
    X = X.unsqueeze(1)  
    class_number = 4

    model = gpt4ts(class_number)
    lightning_model = LitModel_finetune(model)


    trainer = pl.Trainer(
        devices='auto',
        accelerator="cpu",
        strategy=SingleDeviceStrategy(device="cpu"),
        # auto_select_gpus=True,
        benchmark=True,
        enable_checkpointing=True,
    )


    # test the model
    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path=ckpt_path, dataloaders= X
    )[0]
    return pretrain_result

def detspike(file_path):
     # 替换为模型权重实际位置
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fine_tuning', 'spike.ckpt')
    # 替换为.edf文件路径  
    # file_path = "aaaaafoe_00000001.edf"
    batch_size = 512
    try:
        samples = divide_into_samples(file_path,sample_length=5)
    except Exception as e:
        print("{通道不匹配，缺乏所需通道}")
        sys.exit(1)  # 终止程序运行，返回错误码1

    # 获取时间戳
    # 通过定义法计算每个样本的分类
    classifications = []
    # 将 samples 按照 batch size 进行分组
    num_batches = len(samples) // batch_size + (1 if len(samples) % batch_size!= 0 else 0)
    classifications = []
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(samples))
        batch_samples = samples[start:end]
        pretrain_result = supervised(batch_samples, ckpt_path)


    # 定义类别映射
    category_mapping = {
        0: 'normal',
        1: 'SPSW',
        2: 'GPED',
        3: 'PLED'
    }

    # 读取csv文件
    with open('classifications_spike.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        result = [int(row[0]) for row in reader]  # 读取分类结果

    # 平滑处理：确保值连续输出（如有必要）
    def smooth_classifications(result):
        smoothed_result = []
        current_category = result[0]
        duration = 0
        for i in range(len(result)):
            if result[i] == current_category:
                duration += 5  # 每个值代表5秒
            else:
                smoothed_result.append((current_category, duration))
                current_category = result[i]
                duration = 5  # 重置持续时间
        smoothed_result.append((current_category, duration))  # 添加最后一段
        return smoothed_result

    # 获取平滑后的结果
    smoothed_result = smooth_classifications(result)

    # 打印总时长和每个类别的持续时间
    total_duration = len(result) * 5  # 每个值代表5秒
    category_0_duration = 0
    output_text=''
    # print(f"总时长: {total_duration :.2f} 秒 ")
    output_text += f"总时长: {total_duration :.2f} 秒\n"
    for category, duration in smoothed_result:
        # print(f"类别: {category_mapping[category]}, 时长: {duration } 秒")
        output_text += f"类别: {category_mapping[category]}, 时长: {duration } 秒\n"

        if category == 0:
            category_0_duration += duration

    # 计算类别 0（"normal"）占比
    category_0_percentage = (category_0_duration / total_duration) * 100
    # print(f"类别 'normal' 占总时长的比例: {category_0_percentage:.2f}%")
    output_text += f"类别 'normal' 占总时长的比例: {category_0_percentage:.2f}%\n"
    output_text += f"癫痫波检测结束！\n"


    os.remove('classifications_spike.csv')
    return output_text

def batch_test(file_dir):
    pretrain_result=''
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.endswith('.edf'):
                file_path = os.path.join(root, file)
                # 提取.edf 前的数字作为键
                key = file.split('.')[0]
                pretrain_result += f"当前检测文件: {key} \n"
                # print(f"当前检测文件: {key}")
                pretrain_result += detspike(file_path)
    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write(pretrain_result)
    return pretrain_result
if __name__ == "__main__":


    # # 示例用法
    file_path = "20370.edf"

    result = detspike(file_path)

    