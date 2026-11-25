"""FFT 频谱分析。

把时域音频块转换为对数频率轴上的柱状电平（0~1），并做：
- 按 FFT bin 索引映射频段，保证每根柱至少覆盖一个频点，避免低频空柱；
- AGC 自动增益 + 动态范围压缩，避免大音量音乐所有柱子顶到上限；
- 时间平滑与峰值帽保持。
"""

from __future__ import annotations

import numpy as np


class SpectrumAnalyzer:
    def __init__(
        self,
        n_bars: int = 64,
        n_fft: int = 8192,
        freq_min: float = 20.0,
        freq_max: float = 18000.0,
        decay: float = 0.72,
        peak_decay: float = 0.02,
        dyn_range_db: float = 48.0,
        gamma: float = 0.65,
        ceiling: float = 0.88,
        agc_attack: float = 0.9,
        agc_release: float = 0.03,
        noise_floor_db: float = -65.0,
    ):
        self.n_bars = n_bars
        self.n_fft = n_fft
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.decay = decay
        self.peak_decay = peak_decay
        self.dyn_range_db = dyn_range_db
        self.gamma = gamma
        self.ceiling = ceiling
        self.agc_attack = agc_attack
        self.agc_release = agc_release
        self.noise_floor_db = noise_floor_db

        self._window = np.hanning(n_fft)
        self._bin_freq: np.ndarray | None = None
        self._bin_lo: np.ndarray | None = None
        self._bin_hi: np.ndarray | None = None
        self._edges = np.logspace(
            np.log10(freq_min), np.log10(freq_max), n_bars + 1
        )

        self.levels = np.zeros(n_bars)
        self.peaks = np.zeros(n_bars)
        self.last_rms = 0.0
        #: AGC 跟踪的近期峰值电平（dB）
        self.ref_db = -30.0

    def configure_rate(self, rate: int) -> None:
        """根据音频设备采样率重建 FFT 频率轴与每柱的 bin 索引区间。"""
        self._bin_freq = np.fft.rfftfreq(self.n_fft, d=1.0 / rate)
        bin_res = rate / self.n_fft
        n_bins = len(self._bin_freq)

        lo = np.zeros(self.n_bars, dtype=int)
        hi = np.zeros(self.n_bars, dtype=int)
        prev = 1  # 跳过直流 bin 0
        for i in range(self.n_bars):
            f0, f1 = self._edges[i], self._edges[i + 1]
            a = max(prev, int(np.floor(f0 / bin_res)))
            b = max(a + 1, int(np.ceil(f1 / bin_res)))
            a = min(a, n_bins - 1)
            b = min(b, n_bins)
            lo[i], hi[i] = a, b
            prev = a
        self._bin_lo, self._bin_hi = lo, hi

    def analyze(self, samples: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """分析一块音频，返回 (当前电平, 峰值帽)，取值均为 0~1。"""
        if samples is None or len(samples) == 0 or self._bin_freq is None:
            return self.levels, self.peaks

        self.last_rms = float(np.sqrt(np.mean(samples ** 2)))

        if len(samples) < self.n_fft:
            x = np.zeros(self.n_fft, dtype=np.float32)
            x[: len(samples)] = samples
        else:
            x = samples[-self.n_fft:]

        x = x * self._window
        spec = np.abs(np.fft.rfft(x))
        mag = spec / (self.n_fft / 2.0)
        db = 20.0 * np.log10(mag + 1e-7)

        # ---------- AGC：跟踪近期峰值，动态抬高/压低参考电平 ----------
        frame_peak = float(np.max(db[1:]))  # 跳过直流
        if self.last_rms > 1e-4:
            if frame_peak > self.ref_db:
                self.ref_db += (frame_peak - self.ref_db) * self.agc_attack
            else:
                self.ref_db += (frame_peak - self.ref_db) * self.agc_release
        self.ref_db = float(np.clip(self.ref_db, -60.0, -3.0))
        floor_db = self.ref_db - self.dyn_range_db

        # ---------- 逐柱取频段能量 ----------
        raw = np.zeros(self.n_bars)
        for i in range(self.n_bars):
            seg = db[self._bin_lo[i] : self._bin_hi[i]]
            seg = seg[seg > self.noise_floor_db]
            if seg.size:
                raw[i] = np.clip(
                    (float(np.max(seg)) - floor_db) / self.dyn_range_db, 0.0, 1.0
                )

        # 伽马压缩压低顶尖、抬升弱柱；ceiling 给最强柱留出顶部余量
        raw = np.power(raw, self.gamma) * self.ceiling

        # 孤立空柱用左右邻柱插值兜底（不填补整段静音）
        raw = self._fill_gaps(raw)
        raw = np.clip(raw, 0.0, 1.0)

        # 上升立即跟随，下降按衰减系数平滑，避免柱子抖动
        self.levels = np.maximum(raw, self.levels * self.decay)
        # 峰值帽：被新高刷新，否则缓慢下落
        self.peaks = np.where(
            self.levels >= self.peaks, self.levels, self.peaks - self.peak_decay
        )
        self.peaks = np.clip(self.peaks, 0.0, 1.0)
        return self.levels, self.peaks

    def _fill_gaps(self, raw: np.ndarray) -> np.ndarray:
        """对左右都有能量的孤立零值段做线性插值，避免低频出现空缺柱。"""
        out = raw.copy()
        n = len(out)
        i = 0
        while i < n:
            if out[i] != 0.0:
                i += 1
                continue
            j = i
            while j < n and out[j] == 0.0:
                j += 1
            left = out[i - 1] if i > 0 else 0.0
            right = out[j] if j < n else 0.0
            if left > 0.0 and right > 0.0:
                for k in range(i, j):
                    out[k] = left + (right - left) * (k - i + 1) / (j - i + 1)
            i = j
        return out
