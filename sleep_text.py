# sleep_analysis.py - 简化的睡眠分析代码
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.strategies import SingleDeviceStrategy
import os
import csv
import sys
import mne
from scipy.signal import welch
from scipy.integrate import simpson
from model import gpt4ts


# ==================== 第一部分：数据预处理 ====================

def divide_into_samples(file_path, sample_length=30):
    """划分数据为30秒不重叠的样本"""
    raw_data = mne.io.read_raw_edf(file_path, preload=True)
    sfreq = raw_data.info['sfreq']
    sample_points = int(sample_length * sfreq)
    
    n_samples = int(raw_data.n_times / sample_points)
    keywords = ['Fpz', 'Pz', 'Cz', 'Oz']
    
    all_channel_names = raw_data.ch_names
    selected_channels = [ch for ch in all_channel_names if any(keyword.lower() in ch.lower() for keyword in keywords)]
    selected_channels = selected_channels[:2]
    
    if not selected_channels:
        raise ValueError(
            f"No channels matched the keywords: {keywords}. "
            f"Available channels: {all_channel_names}"
        )
    
    raw_data.pick_channels(selected_channels)
    
    samples = []
    for i in range(n_samples):
        start = i * sample_points
        end = (i + 1) * sample_points
        sample = raw_data.get_data(start=start, stop=end)
        samples.append(sample)
    
    return np.array(samples), raw_data, sfreq


# ==================== 第二部分：睡眠分期模型 ====================

class LitModel_finetune(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def test_step(self, batch, batch_idx):
        X = batch
        with torch.no_grad():
            convScore = self.model(X)
            step_result = convScore.cpu().numpy()
        return step_result

    def test_epoch_end(self, test_step_outputs):
        result = []
        for out in test_step_outputs:
            result.append(np.argmax(out[0], axis=0))
        result = np.array(result)
        
        with open('classifications.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for value in result:
                row = [value]
                writer.writerow(row)
        return result


def supervised(X, ckpt_path):
    mean = np.mean(X, axis=-1, keepdims=True)
    std = np.std(X, axis=-1, keepdims=True)
    std = np.maximum(std, 1e-8)
    X = (X - mean) / std
    X = torch.FloatTensor(X)
    X = X.unsqueeze(1)
    class_number = 5

    model = gpt4ts(class_number)
    lightning_model = LitModel_finetune(model)

    trainer = pl.Trainer(
        devices="auto",
        accelerator="cpu",
        strategy=SingleDeviceStrategy(device="cpu"),
        benchmark=True,
        enable_checkpointing=True,
    )

    pretrain_result = trainer.test(
        model=lightning_model, ckpt_path=ckpt_path, dataloaders=X
    )[0]
    return pretrain_result


def smooth_classifications(result, window_size=20):
    """平滑处理分类结果"""
    smoothed_result = []
    for i in range(len(result)):
        window = result[max(0, i - window_size//2):min(len(result), i + window_size//2 + 1)]
        most_common = max(set(window), key=window.count)
        smoothed_result.append(most_common)
    return smoothed_result


# ==================== 第三部分：脑电指标计算 ====================

class BrainMetricsCalculator:
    """脑电指标计算器 - 6个核心指标"""
    
    def __init__(self, raw_data, sfreq, sleep_stages, sample_length=30):
        self.raw_data = raw_data
        self.sfreq = sfreq
        self.sleep_stages = sleep_stages
        self.sample_points = int(sample_length * sfreq)
        self.channel_data = raw_data.get_data()
        self.channel_names = raw_data.ch_names
        
        # 选择关键通道
        self.channels = self._select_channels()
        
    def _select_channels(self):
        """选择分析所需的通道，支持10-20系统最近邻自适应"""
        # 标准10-20系统通道位置映射 (近似坐标)
        # 使用简化的2D坐标 (x, y) 其中y表示前后位置，x表示左右位置
        standard_positions = {
            # 额叶区域
            'Fp1': (-1, 3), 'Fp2': (1, 3), 'Fpz': (0, 3),
            'F7': (-2, 2), 'F3': (-1, 2), 'Fz': (0, 2), 
            'F4': (1, 2), 'F8': (2, 2),
            # 中央区域
            'T7': (-2, 1), 'C3': (-1, 1), 'Cz': (0, 1),
            'C4': (1, 1), 'T8': (2, 1),
            # 顶叶区域
            'P7': (-2, 0), 'P3': (-1, 0), 'Pz': (0, 0),
            'P4': (1, 0), 'P8': (2, 0),
            # 枕叶区域
            'O1': (-1, -1), 'Oz': (0, -1), 'O2': (1, -1),
        }
        
        # 定义目标通道组及其理想位置
        target_groups = {
            'sleep': {
                'channels': ['Fpz', 'Cz', 'Pz', 'Oz'],
                'positions': [(0, 3), (0, 1), (0, 0), (0, -1)]
            },
            'focus': {
                'channels': ['Fpz', 'Cz', 'Pz', 'Oz'],
                'positions': [(0, 3), (0, 1), (0, 0), (0, -1)]
            }
        }
        
        selected = {'sleep': [], 'focus': []}
        
        # 为每个目标组选择通道
        for group_name, group_info in target_groups.items():
            target_channels = group_info['channels']
            target_positions = group_info['positions']
            
            # 首先尝试精确匹配
            found_channels = []
            for target_ch in target_channels:
                matched = False
                for ch in self.channel_names:
                    if target_ch.lower() in ch.lower():
                        found_channels.append(ch)
                        matched = True
                        break
                if not matched:
                    found_channels.append(None)  # 标记未找到
            
            # 如果有缺失的通道，使用最近邻替换
            if None in found_channels:
                # 获取所有可用通道的位置
                available_positions = {}
                for ch in self.channel_names:
                    # 尝试从标准位置中查找
                    for std_ch, pos in standard_positions.items():
                        if std_ch.lower() in ch.lower():
                            available_positions[ch] = pos
                            break
                
                # 如果有可用通道位置信息
                if available_positions:
                    # 对每个缺失的通道进行替换
                    for i, (target_ch, target_pos) in enumerate(zip(target_channels, target_positions)):
                        if found_channels[i] is None:
                            # 计算最近邻
                            nearest_ch = self._find_nearest_channel(
                                target_pos, 
                                available_positions, 
                                found_channels  # 排除已选通道
                            )
                            if nearest_ch:
                                found_channels[i] = nearest_ch
                                # 从可用位置中移除已选的通道，避免重复
                                if nearest_ch in available_positions:
                                    del available_positions[nearest_ch]
            
            # 过滤掉None值（如果实在找不到）
            selected[group_name] = [ch for ch in found_channels if ch is not None]
            
            # 如果选中的通道少于需求，补充其他通道
            if len(selected[group_name]) < len(target_channels):
                remaining = [ch for ch in self.channel_names 
                            if ch not in selected[group_name]]
                needed = len(target_channels) - len(selected[group_name])
                selected[group_name].extend(remaining[:needed])
        
        return selected

    def _find_nearest_channel(self, target_pos, available_positions, exclude_channels):
        """找到最近邻通道"""
        from math import sqrt
        
        min_distance = float('inf')
        nearest_ch = None
        
        for ch, pos in available_positions.items():
            # 排除已选择的通道
            if ch in exclude_channels:
                continue
                
            # 计算欧几里得距离
            distance = sqrt((pos[0] - target_pos[0])**2 + 
                        (pos[1] - target_pos[1])**2)
            
            if distance < min_distance:
                min_distance = distance
                nearest_ch = ch
        
        return nearest_ch
    
    def _get_channel_data(self, channel_name):
        """获取指定通道的数据"""
        for i, name in enumerate(self.channel_names):
            if channel_name.lower() in name.lower():
                return self.channel_data[i]
        return None
    
    def _calculate_band_power(self, signal, band):
        """计算指定频段的功率"""
        freqs, psd = welch(signal, self.sfreq, nperseg=min(len(signal), self.sfreq*2))
        idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
        if np.sum(idx_band) == 0:
            return 0
        return simpson(psd[idx_band], freqs[idx_band])
    
    def _calculate_band_ratio(self, signal, band1, band2):
        """计算两个频段的功率比"""
        p1 = self._calculate_band_power(signal, band1)
        p2 = self._calculate_band_power(signal, band2)
        if p2 < 1e-10:
            return 0
        return p1 / p2

    # ----- 指标1: 睡眠时长 -----
    def calculate_sleep_duration(self):
        """睡眠时长：所有睡眠期（非W期）的总时间，单位：分钟"""
        sleep_mask = np.array(self.sleep_stages) != 0
        return np.sum(sleep_mask) * 30 / 60

    # ----- 指标2: 入睡困难 -----
    def calculate_sleep_latency(self):
        """入睡困难：入睡潜伏期，>30分钟为入睡困难，单位：分钟"""
        for i, stage in enumerate(self.sleep_stages):
            if stage != 0:
                latency = i * 30 / 60
                return {
                    'latency_minutes': latency,
                    'has_difficulty': latency > 30,
                    'status': '入睡困难' if latency > 30 else '正常'
                }
        return {
            'latency_minutes': None,
            'has_difficulty': True,
            'status': '未检测到入睡'
        }

    # ----- 指标3: 易醒早醒 -----
    def calculate_arousal_metrics(self):
        """易醒早醒：觉醒指数和WASO"""
        # 计算觉醒指数
        arousal_count = 0
        i = 0
        while i < len(self.sleep_stages):
            if self.sleep_stages[i] == 0:
                start = i
                while i < len(self.sleep_stages) and self.sleep_stages[i] == 0:
                    i += 1
                duration = (i - start) * 30
                if duration < 180:
                    arousal_count += 1
            else:
                i += 1
        
        total_sleep_hours = self.calculate_sleep_duration() / 60
        arousal_index = arousal_count / total_sleep_hours if total_sleep_hours > 0 else 0
        
        # WASO
        first_sleep_idx = None
        for i, stage in enumerate(self.sleep_stages):
            if stage != 0:
                first_sleep_idx = i
                break
        
        waso_minutes = 0
        if first_sleep_idx is not None:
            waso_seconds = 0
            for i in range(first_sleep_idx, len(self.sleep_stages)):
                if self.sleep_stages[i] == 0:
                    waso_seconds += 30
            waso_minutes = waso_seconds / 60
        
        return {
            'arousal_index': arousal_index,
            'waso_minutes': waso_minutes,
            'is_arousal': arousal_index > 15,
            'status': '易醒' if arousal_index > 15 else '正常'
        }

    # ----- 指标4: 是否多梦 -----
    def calculate_dream_metrics(self):
        """是否多梦：通过REM睡眠推断"""
        total_sleep = np.sum(np.array(self.sleep_stages) != 0)
        rem_sleep = np.sum(np.array(self.sleep_stages) == 4)
        rem_percentage = (rem_sleep / total_sleep * 100) if total_sleep > 0 else 0
        
        # REM潜伏期
        rem_latency = None
        first_rem_idx = None
        for i, stage in enumerate(self.sleep_stages):
            if stage == 4:
                first_rem_idx = i
                break
        if first_rem_idx is not None:
            sleep_latency = self.calculate_sleep_latency()
            if sleep_latency['latency_minutes'] is not None:
                rem_latency = first_rem_idx * 30 / 60 - sleep_latency['latency_minutes']
            else:
                rem_latency = first_rem_idx * 30 / 60
        
        is_dream = rem_percentage > 25
        if not is_dream and rem_latency is not None:
            is_dream = rem_latency < 60
        
        return {
            'rem_percentage': rem_percentage,
            'rem_latency': rem_latency,
            'is_dream': is_dream,
            'status': '可能多梦' if is_dream else '正常'
        }

    # ----- 指标5: 疲劳程度 -----
    def calculate_fatigue_level(self):
        """疲劳程度：θ波与α波功率比"""
        focus_channel = self.channels['focus'][0]
        signal = self._get_channel_data(focus_channel)
        if signal is None:
            signal = self.channel_data[0]
        
        theta_alpha_ratio = self._calculate_band_ratio(signal, (4, 8), (8, 13))
        
        fatigue_level = min(theta_alpha_ratio / 2.0, 1.0)
        
        if fatigue_level < 0.3:
            status = '精力充沛'
        elif fatigue_level < 0.6:
            status = '轻度疲劳'
        elif fatigue_level < 0.8:
            status = '中度疲劳'
        else:
            status = '严重疲劳'
        
        return {
            'theta_alpha_ratio': theta_alpha_ratio,
            'fatigue_index': fatigue_level,
            'status': status
        }

    # ----- 指标6: 专注能力 -----
    def calculate_focus_level(self):
        """专注能力：额叶θ波功率增强，α波抑制"""
        focus_channel = self.channels['focus'][0] if self.channels['focus'] else self.channel_names[0]
        signal = self._get_channel_data(focus_channel)
        if signal is None:
            signal = self.channel_data[0]
        
        theta_power = self._calculate_band_power(signal, (4, 8))
        alpha_power = self._calculate_band_power(signal, (8, 13))
        
        focus_index = theta_power / (theta_power + alpha_power + 1e-8)
        alpha_suppression = 1 - min(alpha_power / (theta_power + 1e-8), 1)
        
        if focus_index > 0.6:
            status = '高度专注'
            level = 4
        elif focus_index > 0.45:
            status = '良好专注'
            level = 3
        elif focus_index > 0.3:
            status = '一般专注'
            level = 2
        else:
            status = '注意力分散'
            level = 1
        
        return {
            'focus_index': focus_index,
            'alpha_suppression': alpha_suppression,
            'level': level,
            'status': status
        }

    # ----- 获取全部6个指标 -----
    def get_all_metrics(self, expected_wakeup_hour=None):
        """获取全部6个核心指标"""
        # 1. 睡眠时长
        sleep_duration = self.calculate_sleep_duration()
        
        # 2. 入睡困难
        sleep_latency = self.calculate_sleep_latency()
        
        # 3. 易醒早醒
        arousal_metrics = self.calculate_arousal_metrics()
        
        # 早醒判断
        early_wake = False
        if expected_wakeup_hour is not None:
            last_wake_idx = None
            for i in range(len(self.sleep_stages)-1, -1, -1):
                if self.sleep_stages[i] == 0:
                    last_wake_idx = i
                    break
            if last_wake_idx is not None:
                wake_time = last_wake_idx * 30 / 60
                expected_wake_minutes = expected_wakeup_hour * 60
                early_wake = (expected_wake_minutes - wake_time) > 30
        arousal_metrics['early_wake'] = early_wake
        
        # 4. 是否多梦
        dream_metrics = self.calculate_dream_metrics()
        
        # 5. 疲劳程度
        fatigue_metrics = self.calculate_fatigue_level()
        
        # 6. 专注能力
        focus_metrics = self.calculate_focus_level()
        
        return {
            'sleep_duration': sleep_duration,
            'sleep_latency': sleep_latency,
            'arousal_metrics': arousal_metrics,
            'dream_metrics': dream_metrics,
            'fatigue_metrics': fatigue_metrics,
            'focus_metrics': focus_metrics
        }


# ==================== 第四部分：主函数 - 生成完整报告 ====================

def analyze_sleep(file_path, expected_wakeup_hour=None):
    """
    主函数：返回包含睡眠分期和6个脑电指标的完整报告
    
    参数:
    file_path: EDF文件路径
    expected_wakeup_hour: 预期起床时间（小时，0-24），用于早醒判断
    """
    
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fine_tuning', 'gpt4ts-sleep.ckpt')
    batch_size = 512
    
    try:
        samples, raw_data, sfreq = divide_into_samples(file_path, sample_length=30)
    except Exception as e:
        return f"❌ 通道不匹配，缺乏所需通道: {str(e)}"
    
    # 执行分类
    num_batches = len(samples) // batch_size + (1 if len(samples) % batch_size != 0 else 0)
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(samples))
        batch_samples = samples[start:end]
        pretrain_result = supervised(batch_samples, ckpt_path)

    # 读取分类结果
    sleep_stage_mapping = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}
    
    with open('classifications.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        result = [int(row[0]) for row in reader]
    
    # 平滑处理
    smoothed_result = smooth_classifications(result)
    
    # 删除临时文件
    os.remove('classifications.csv')
    
    # ===== 生成完整报告 =====
    report_text = ""
    report_text += "="*70 + "\n"
    report_text += "                    睡眠分析完整报告\n"
    report_text += "="*70 + "\n\n"
    
    # ----- 第一部分：睡眠分期 -----
    report_text += "【一、睡眠分期】\n"
    report_text += "-"*50 + "\n"
    
    total_duration = len(smoothed_result) * 30
    report_text += f"总记录时长: {total_duration / 60:.2f} 分钟\n\n"
    
    current_stage = smoothed_result[0]
    stage_duration = 0
    
    for i in range(len(smoothed_result)):
        if smoothed_result[i] == current_stage:
            stage_duration += 30
        else:
            report_text += f"  {sleep_stage_mapping[current_stage]}: {stage_duration / 60:.2f} 分钟\n"
            current_stage = smoothed_result[i]
            stage_duration = 30
    
    if stage_duration > 0:
        report_text += f"  {sleep_stage_mapping[current_stage]}: {stage_duration / 60:.2f} 分钟\n"
    
    report_text += "\n"
    
    # ----- 第二部分：脑电指标 -----
    report_text += "【二、脑电指标分析】\n"
    report_text += "-"*50 + "\n"
    
    # 重新加载完整数据用于指标计算
    raw_data_full = mne.io.read_raw_edf(file_path, preload=True)
    metrics_calc = BrainMetricsCalculator(raw_data_full, raw_data_full.info['sfreq'], smoothed_result)
    metrics = metrics_calc.get_all_metrics(expected_wakeup_hour)
    
    # 指标1: 睡眠时长
    report_text += f"\n【指标1】睡眠时长\n"
    report_text += f"  总睡眠时间: {metrics['sleep_duration']:.2f} 分钟 ({metrics['sleep_duration']/60:.2f} 小时)\n"
    if metrics['sleep_duration'] < 420:
        report_text += f"  ⚠️ 睡眠时长不足（建议7-8小时）\n"
    elif metrics['sleep_duration'] > 540:
        report_text += f"  ⚠️ 睡眠时间偏长\n"
    else:
        report_text += f"  ✅ 睡眠时长正常\n"
    
    # 指标2: 入睡困难
    report_text += f"\n【指标2】入睡困难\n"
    latency = metrics['sleep_latency']
    if latency['latency_minutes'] is not None:
        report_text += f"  入睡潜伏期: {latency['latency_minutes']:.2f} 分钟\n"
        if latency['has_difficulty']:
            report_text += f"  ⚠️ 存在入睡困难（潜伏期 > 30分钟）\n"
        else:
            report_text += f"  ✅ 入睡正常\n"
    else:
        report_text += f"  ⚠️ 未检测到入睡\n"
    
    # 指标3: 易醒早醒
    report_text += f"\n【指标3】易醒早醒\n"
    arousal = metrics['arousal_metrics']
    report_text += f"  觉醒指数: {arousal['arousal_index']:.1f} 次/小时\n"
    report_text += f"  入睡后清醒时间(WASO): {arousal['waso_minutes']:.2f} 分钟\n"
    if arousal['is_arousal']:
        report_text += f"  ⚠️ 存在易醒问题（觉醒指数 > 15）\n"
    else:
        report_text += f"  ✅ 无易醒问题\n"
    if arousal.get('early_wake', False):
        report_text += f"  ⚠️ 存在早醒问题（比预期早醒30分钟以上）\n"
    
    # 指标4: 是否多梦
    report_text += f"\n【指标4】是否多梦\n"
    dream = metrics['dream_metrics']
    report_text += f"  REM睡眠占比: {dream['rem_percentage']:.1f}%\n"
    if dream['rem_latency'] is not None:
        report_text += f"  REM潜伏期: {dream['rem_latency']:.2f} 分钟\n"
    if dream['is_dream']:
        report_text += f"  💭 可能多梦（REM占比 > 25% 或 REM潜伏期缩短）\n"
    else:
        report_text += f"  ✅ 无多梦迹象\n"
    
    # 指标5: 疲劳程度
    report_text += f"\n【指标5】疲劳程度\n"
    fatigue = metrics['fatigue_metrics']
    report_text += f"  θ/α比值: {fatigue['theta_alpha_ratio']:.3f}\n"
    report_text += f"  疲劳指数: {fatigue['fatigue_index']:.2f}\n"
    report_text += f"  状态: {fatigue['status']}\n"
    
    # 指标6: 专注能力
    report_text += f"\n【指标6】专注能力\n"
    focus = metrics['focus_metrics']
    report_text += f"  专注指数: {focus['focus_index']:.3f}\n"
    report_text += f"  α波抑制: {focus['alpha_suppression']:.2f}\n"
    report_text += f"  专注等级: {focus['level']} 级（{focus['status']}）\n"
    
    # ----- 第三部分：综合建议 -----
    report_text += "\n" + "="*70 + "\n"
    report_text += "【三、综合建议】\n"
    report_text += "-"*50 + "\n"
    
    suggestions = []
    
    if metrics['sleep_duration'] < 420:
        suggestions.append("• 睡眠时长不足，建议保证7-8小时睡眠")
    elif metrics['sleep_duration'] > 540:
        suggestions.append("• 睡眠时间偏长，可能存在睡眠质量问题")
    
    if latency['has_difficulty']:
        suggestions.append("• 存在入睡困难，建议睡前放松、避免咖啡因和电子产品")
    
    if arousal['is_arousal']:
        suggestions.append("• 睡眠碎片化，建议改善睡眠环境、减少夜间干扰")
    if arousal.get('early_wake', False):
        suggestions.append("• 存在早醒问题，建议调整作息时间")
    
    if dream['is_dream']:
        suggestions.append("• 可能多梦，建议减少压力、避免睡前过饱")
    
    if fatigue['fatigue_index'] > 0.6:
        suggestions.append("• 疲劳程度较高，建议适当休息和放松")
    
    if focus['focus_index'] < 0.3:
        suggestions.append("• 注意力较分散，建议进行专注力训练")
    
    if not suggestions:
        suggestions.append("• 各项指标良好，继续保持健康作息")
    
    report_text += "\n".join(suggestions) + "\n"
    report_text += "="*70 + "\n"
    report_text += "报告生成时间: " + str(np.datetime64('now')) + "\n"
    
    return report_text


# ==================== 主程序入口 ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='睡眠分析工具')
    parser.add_argument('--input', '-i', type=str, required=True, help='EDF文件路径')
    parser.add_argument('--wakeup', '-w', type=float, default=7, help='预期起床时间 (小时, 0-24)')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.input):
        print(f"❌ 文件不存在: {args.input}")
        sys.exit(1)
    
    # 执行分析
    print(f"🔍 正在分析文件: {args.input}")
    print("-" * 70)
    
    report = analyze_sleep(args.input, expected_wakeup_hour=args.wakeup)
    
    # 打印报告
    print(report)
    
    # 可选：保存报告到文件
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    output_file = f"{base_name}_sleep_report.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n📄 报告已保存到: {output_file}")