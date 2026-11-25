"""程序入口：启动实时音乐频谱界面。"""

from __future__ import annotations

import tkinter as tk

from music_spectrum.gui import SpectrumApp


def main() -> None:
    root = tk.Tk()
    SpectrumApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
