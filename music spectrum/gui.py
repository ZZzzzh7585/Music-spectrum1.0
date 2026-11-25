"""tkinter 实时频谱界面。"""

from __future__ import annotations

import colorsys
import tkinter as tk

import numpy as np

from .audio_capture import AudioCapture
from .spectrum import SpectrumAnalyzer

BG = "#0b0f14"
FG = "#c9d1d9"
MUTED = "#6e7681"
BAR_GAP = 3
FPS_MS = 30
SILENCE_LEVEL = 0.03
SILENCE_RMS = 0.004
SILENCE_FRAMES = 130  # 约 3.9 秒无声音后尝试切换设备


class SpectrumApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("实时音乐频谱")
        root.geometry("960x520")
        root.minsize(640, 360)
        root.configure(bg=BG)

        self.status = tk.Label(
            root,
            text="正在启动音频捕获…",
            anchor="w",
            bg=BG,
            fg=FG,
            font=("Microsoft YaHei UI", 10),
        )
        self.status.pack(fill="x", padx=12, pady=(8, 2))

        self.canvas = tk.Canvas(root, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.canvas.bind("<Configure>", lambda _e: self._draw())

        self.capture = AudioCapture(chunk=4096)
        self.analyzer = SpectrumAnalyzer(n_bars=64)
        self._bar_colors = self._make_colors(self.analyzer.n_bars)
        self._silent_frames = 0

        ok = self.capture.start()
        if ok:
            self.analyzer.configure_rate(self.capture.rate)
            self.status.config(
                text=(
                    f"输入：{self.capture.device_name}"
                    f"　|　采样率：{self.capture.rate} Hz"
                    f"　|　播放任意音乐即可看到频谱"
                )
            )
        else:
            self.status.config(text=self.capture.error or "音频设备启动失败")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(FPS_MS, self._tick)

    # ---------- 颜色 ----------
    @staticmethod
    def _make_colors(n: int) -> list[str]:
        """低频蓝青 -> 高频红 的渐变柱色。"""
        colors = []
        for i in range(n):
            t = i / max(1, n - 1)
            hue = 0.72 * (1.0 - t)
            r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.9)
            colors.append(
                f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"
            )
        return colors

    # ---------- 帧循环 ----------
    def _tick(self) -> None:
        samples = self.capture.get_latest()
        self.analyzer.analyze(samples)

        if self.analyzer.last_rms < SILENCE_RMS:
            self._silent_frames += 1
        else:
            self._silent_frames = 0

        # 持续静音且还有其他输出设备候选时，自动切换尝试
        if self._silent_frames > SILENCE_FRAMES and self.capture.has_next:
            if self.capture.reopen_next():
                self.analyzer.configure_rate(self.capture.rate)
                self.analyzer.levels = np.zeros(self.analyzer.n_bars)
                self.analyzer.peaks = np.zeros(self.analyzer.n_bars)
                self.status.config(
                    text=f"未在当前设备检测到声音，已切换到：{self.capture.device_name}"
                )
            self._silent_frames = 0

        self._draw()
        self.root.after(FPS_MS, self._tick)

    # ---------- 绘制 ----------
    def _draw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 50 or h < 50:
            return

        n = self.analyzer.n_bars
        margin_x = 16
        margin_bottom = 14
        margin_top = 18
        gap = BAR_GAP
        bar_w = (w - margin_x * 2 - gap * (n - 1)) / n
        base_y = h - margin_bottom
        avail_h = base_y - margin_top

        if float(np.max(self.analyzer.levels)) < SILENCE_LEVEL:
            canvas.create_text(
                w / 2,
                h / 2,
                text="正在等待音乐播放…\n（打开本地歌曲或 QQ 音乐等播放器即可）",
                fill=MUTED,
                font=("Microsoft YaHei UI", 14),
                justify="center",
            )
            return

        levels = self.analyzer.levels
        peaks = self.analyzer.peaks
        for i in range(n):
            x0 = margin_x + i * (bar_w + gap)
            x1 = x0 + bar_w
            lvl = float(levels[i])
            y_top = base_y - lvl * avail_h
            canvas.create_rectangle(
                x0, y_top, x1, base_y, fill=self._bar_colors[i], outline=""
            )
            pk = float(peaks[i])
            if pk > 0.02:
                py = base_y - pk * avail_h
                canvas.create_rectangle(x0, py - 3, x1, py, fill="#ffffff", outline="")

    # ---------- 退出 ----------
    def _on_close(self) -> None:
        self.capture.stop()
        self.root.destroy()
