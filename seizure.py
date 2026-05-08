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
    # selected_channels = [ch for ch in all_channel_names if any(keyword in ch for keyword in keywords)]
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
        self.threshold = 0.4153   # 4153 3144


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
            # self.log("result", float(result), sync_dist=True)

        if (
             len(result)!=0
        ):  # to prevent all 0 or all 1 and raise the AUROC error
            # print(result)
            y_pred = result.copy()
            y_pred[y_pred >= self.threshold] = 1
            y_pred[y_pred < self.threshold] = 0
            # print(y_pred)
            self.log("检测结果", int(y_pred.copy()), sync_dist=True)

            return y_pred



def supervised(file_path):
    try:
        X = edf_eeg(file_path)
    except Exception as e:
        print("{通道不匹配，缺乏所需通道}")
        sys.exit(1)  # 终止程序运行，返回错误码1
    # X = file_path
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fine_tuning', 'seizure.ckpt')

    mean = np.mean(X, axis=-1, keepdims=True)
    std = np.std(X, axis=-1, keepdims=True)
    std = np.maximum(std, 1e-8)
    X = (X - mean) / std
    X = torch.FloatTensor(X)
    X = X.unsqueeze(0)  
    X = X.unsqueeze(0)  
    # print("输入：",X.shape)

    model = gpt4ts(1)
    # model = labram_base_patch200_200()

    lightning_model = LitModel_finetune(model)


    trainer = pl.Trainer(
        devices='auto',
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
    print(pretrain_result)
    return pretrain_result
if __name__ == "__main__":

    # # 替换为.edf文件路径  
    # file_path = "20370.edf"
    # # X=edf_eeg(file_path)
    # supervised(file_path)
    def batch_test(file_dir):
        result_dict = {}
        for root, dirs, files in os.walk(file_dir):
            for file in files:
                if file.endswith('.edf'):
                    file_path = os.path.join(root, file)
                    # 提取.edf 前的数字作为键
                    key = file.split('.')[0]
                    # X = edf_eeg(file_path)
                    pretrain_result = supervised(file_path)
                    result_dict[key] = pretrain_result['检测结果']
        return result_dict

    # # 示例用法
    # path=r"D:\EEG数据集\test\edf2\da001ubk.edf"

    # file_path = "D:\EEG数据集\四川省人民医院数据\脑电图分析"
    file_path=r"C:\Users\Administrator\Desktop\临床bench\normal"

    result = batch_test(file_path)
    print(result)

    # 将结果续写到CSV文件
    import csv
    import os

    def append_to_csv(result, csv_file="results_normal.csv"):
        """
        将键值对结果续写到CSV文件
        """
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 如果文件不存在，写入表头
            if not file_exists:
                writer.writerow(["键", "值"])
            
            # 将键值对写入CSV
            for key, value in result.items():
                writer.writerow([key, value])
        
        print(f"数据已成功续写到 {csv_file}")

    append_to_csv(result)
