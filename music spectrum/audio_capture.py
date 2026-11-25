"""系统声音捕获。

通过 Windows WASAPI 环路回放（loopback）设备录制电脑正在播放的声音，
无论是本地播放器还是 QQ 音乐等网页 / 客户端播放的音乐，都能被捕获。
若没有 loopback 设备，则兜底使用“立体声混音（Stereo Mix）”。

启动时按「默认扬声器对应的 loopback → 其他 loopback → 立体声混音」
排出候选列表；运行中若当前设备持续静音，可由界面层调用 reopen_next()
切换到下一个候选设备。
"""

from __future__ import annotations

import threading

import numpy as np
import pyaudiowpatch as pyaudio


class AudioCapture:
    def __init__(self, chunk: int = 2048):
        self.chunk = chunk
        self._pa = pyaudio.PyAudio()
        self._stream = None
        self._thread: threading.Thread | None = None
        self._running = False

        #: 候选设备列表与当前位置
        self.candidates: list[dict] = []
        self._cand_pos = -1

        #: 最新一块单声道 float32 音频
        self.latest: np.ndarray | None = None
        self.device_name = ""
        self.rate = 0
        self.channels = 0
        self.error: str | None = None

    # ---------- 设备枚举 ----------
    def _list_candidates(self) -> None:
        wasapi = self._pa.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_out = self._pa.get_device_info_by_index(wasapi["defaultOutputDevice"])
        out_name = default_out["name"]

        matched: list[dict] = []
        others: list[dict] = []
        stereo: list[dict] = []

        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if not info.get("maxInputChannels", 0):
                continue

            if info.get("isLoopbackDevice"):
                base = info["name"].replace(" [Loopback]", "")
                if base == out_name:
                    matched.append(info)
                else:
                    others.append(info)
            elif "立体声混音" in info["name"] or "Stereo Mix" in info["name"]:
                stereo.append(info)

        self.candidates = matched + others + stereo

    # ---------- 生命周期 ----------
    def start(self) -> bool:
        """启动后台采集线程。成功返回 True，失败时错误信息写入 self.error。"""
        self._list_candidates()
        return self._open_next()

    def _open_next(self) -> bool:
        """打开候选列表中的下一个设备。"""
        self._close_stream()
        self._cand_pos += 1

        if self._cand_pos >= len(self.candidates):
            self.error = "未检测到可用的声音输出，请确认电脑正在播放音乐"
            return False

        device = self.candidates[self._cand_pos]
        try:
            self.device_name = device["name"]
            self.rate = int(device["defaultSampleRate"])
            self.channels = int(device["maxInputChannels"])

            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=self.channels,
                rate=self.rate,
                frames_per_buffer=self.chunk,
                input=True,
                input_device_index=device["index"],
            )
            self._running = True
            self._thread = threading.Thread(
                target=self._read_loop, name="audio-capture", daemon=True
            )
            self._thread.start()
            self.error = None
            return True
        except Exception as exc:  # noqa: BLE001
            self.error = f"打开音频设备失败：{exc}"
            # 当前设备打不开，继续尝试下一个
            return self._open_next()

    @property
    def has_next(self) -> bool:
        return self._cand_pos + 1 < len(self.candidates)

    def reopen_next(self) -> bool:
        """切换到下一个候选设备；没有更多候选时返回 False。"""
        if not self.has_next:
            return False
        return self._open_next()

    def _read_loop(self) -> None:
        while self._running:
            try:
                raw = self._stream.read(self.chunk, exception_on_overflow=False)
            except Exception:
                continue
            samples = np.frombuffer(raw, dtype=np.float32)
            if self.channels > 1:
                samples = samples.reshape(-1, self.channels).mean(axis=1)
            self.latest = samples.copy()

    def get_latest(self) -> np.ndarray | None:
        return self.latest

    def _close_stream(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
        self._stream = None
        self.latest = None

    def stop(self) -> None:
        """停止采集并释放音频设备。关闭窗口时调用，退出后不留后台进程。"""
        self._close_stream()
        self._pa.terminate()
