import mne
import numpy as np
from mne.preprocessing import ICA

class ICAArtifactRemover:
    def __init__(self, l_freq=1.0, h_freq=40.0, resample_sfreq=200,
                 method='fastica', n_components=None, random_state=42, max_iter=800):
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.resample_sfreq = resample_sfreq
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.ica = None

    def fit_transform(self, edf_path):
        """读取EDF，ICA分解 + 各独立成分频段占比统计（脑电为大区间，伪迹为补充）"""
        # 1. 读取数据
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

        # 2. 滤波
        raw.filter(self.l_freq, self.h_freq, fir_design='firwin', verbose=False)

        # 3. 下采样
        if self.resample_sfreq and raw.info["sfreq"] > self.resample_sfreq:
            raw.resample(self.resample_sfreq, verbose=False)

        # 4. ICA 拟合
        ica = ICA(
            n_components=self.n_components,
            method=self.method,
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        ica.fit(raw, verbose=False)

        # 5. 提取独立成分
        sources = ica.get_sources(raw).get_data()   # (n_components, n_times)
        sfreq = raw.info["sfreq"]

        results = {}
        for i, sig in enumerate(sources):
            psd, freqs = mne.time_frequency.psd_array_welch(
                sig, sfreq=sfreq, fmin=0.1, fmax=80, n_fft=1024, verbose=False
            )
            total_power = np.sum(psd)

            # --- 脑电（大区间） ---
            eeg_power    = np.sum(psd[(freqs >= 0.5) & (freqs < 40)])

            # --- 伪迹（小区间） ---
            slow_power   = np.sum(psd[(freqs >= 0.1) & (freqs < 0.5)])   # 极慢波
            ecg_power    = np.sum(psd[(freqs >= 0.8) & (freqs < 1.5)])   # 心电
            eye_power    = np.sum(psd[(freqs >= 0.5) & (freqs < 3)])     # 眼动/眨眼
            emg_power    = np.sum(psd[(freqs >= 30) & (freqs <= 80)])    # 肌电
            line_power   = np.sum(psd[(freqs >= 49) & (freqs <= 51)])    # 工频

            explained = eeg_power + slow_power + ecg_power + eye_power + emg_power + line_power
            noise_power = max(0, total_power - explained)

            # --- 百分比 ---
            results[f"IC{i+1:02d}"] = {
                "脑电(0.5–40Hz)": round(100 * eeg_power / total_power, 2),
                "极慢波(0.1–0.5Hz)": round(100 * slow_power / total_power, 2),
                "心电(~1Hz)": round(100 * ecg_power / total_power, 2),
                "眼动/眨眼(0.5–3Hz)": round(100 * eye_power / total_power, 2),
                "肌电(30–80Hz)": round(100 * emg_power / total_power, 2),
                "工频(49–51Hz)": round(100 * line_power / total_power, 2),
                "噪声": round(100 * noise_power / total_power, 2),
            }

        return results
