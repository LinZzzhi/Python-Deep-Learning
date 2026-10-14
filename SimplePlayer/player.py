# -*- coding: utf-8 -*-
"""
SimplePlayer —— 一个简单的音视频播放器

技术栈：
    - Tkinter          : 图形界面（Python 自带，无需额外安装）
    - ffpyplayer       : 解码与播放。该库**静态内嵌了 FFmpeg**（libav* / swscale /
                         swresample 全部编译进扩展模块），因此本机无需另装 ffmpeg。

界面：深色扁平风格。按钮 / 进度条 / 音量条都是 Canvas 自绘，图标用矢量绘制，
      不依赖任何第三方 UI 库或图标字体。

支持格式：由内嵌 FFmpeg 决定，常见音视频格式均可播放
    - 视频: mp4 / mkv / avi / mov / flv / webm / wmv / ts / m4v / mpg / 3gp ...
    - 音频: mp3 / flac / wav / aac / m4a / ogg / opus / wma / ape / ac3 ...

运行：双击 run.bat，或命令行执行
    .venv\\Scripts\\python.exe player.py [媒体文件]
"""

import os
import sys
import time
import tkinter as tk
from tkinter import filedialog, messagebox

from ffpyplayer.player import MediaPlayer

try:
    # 拖放支持：Tk 本身不支持，靠 tkdnd 扩展（tkinterdnd2 已把扩展打包进 wheel）
    from tkinterdnd2 import DND_FILES, TkinterDnD
    HAS_DND = True
except Exception:                      # 装不上就退化成只能用「打开文件」按钮
    DND_FILES = None
    TkinterDnD = None
    HAS_DND = False

APP_TITLE = "SimplePlayer"

VIDEO_EXTS = ("mp4", "mkv", "avi", "mov", "flv", "webm", "wmv", "ts", "m4v",
              "mpg", "mpeg", "3gp", "rmvb", "vob", "ogv", "m2ts", "asf")
AUDIO_EXTS = ("mp3", "flac", "wav", "aac", "m4a", "ogg", "oga", "opus", "wma",
              "ape", "alac", "amr", "mp2", "ac3", "dts", "mid", "aiff")

# 定位后允许的偏差（秒）：偏差内的帧视为已到达目标位置
SEEK_TOLERANCE = 0.4
# 定位后的最长等待时间（秒），超时则不再丢弃帧（兜底，避免画面一直停着）
SEEK_TIMEOUT = 3.0
# 打开文件后等多久才敢断定「这是纯音频」（视频流信息可能来得比时长晚）
AUDIO_PROBE_DELAY = 1.0

# ---------------------------------------------------------------- 配色 / 字体

C_BG = "#15171c"          # 窗口底色
C_PANEL = "#1b1e25"       # 控制栏底色
C_SEP = "#262b34"         # 分隔线
C_TRACK = "#2b303a"       # 进度/音量轨道
C_TRACK_HL = "#3b4250"    # 轨道悬停
C_ACCENT = "#4f9cf9"      # 主色
C_ACCENT_H = "#6db0ff"    # 主色悬停
C_ACCENT_A = "#3a7fd0"    # 主色按下
C_TEXT = "#e7e9ee"        # 主文字
C_TEXT_DIM = "#828a98"    # 次要文字
C_BTN = "#2c323d"         # 普通按钮
C_BTN_H = "#3a4250"
C_BTN_A = "#252a33"
C_HINT = "#69707d"        # 画面区提示文字

FONT_UI = ("Microsoft YaHei UI", 9)
FONT_TIME = ("Consolas", 10)
FONT_HINT = ("Microsoft YaHei UI", 12)


def fmt_time(seconds):
    """把秒数格式化成 00:00 或 00:00:00 形式。"""
    if seconds is None or seconds < 0:
        return "--:--"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return "%d:%02d:%02d" % (h, m, s)
    return "%02d:%02d" % (m, s)


# ---------------------------------------------------------------- 绘图工具

def round_rect(canvas, x0, y0, x1, y1, r, **kw):
    """在 Canvas 上画圆角矩形（用平滑样条多边形近似，Tk 没有原生圆角矩形）。"""
    r = max(0.0, min(r, (x1 - x0) / 2.0, (y1 - y0) / 2.0))
    pts = [x0 + r, y0, x1 - r, y0, x1, y0, x1, y0 + r, x1, y1 - r, x1, y1,
           x1 - r, y1, x0 + r, y1, x0, y1, x0, y1 - r, x0, y0 + r, x0, y0]
    return canvas.create_polygon(pts, smooth=True, **kw)


def icon_play(c, cx, cy, s, color):
    c.create_polygon(cx - s * 0.30, cy - s * 0.46,
                     cx - s * 0.30, cy + s * 0.46,
                     cx + s * 0.52, cy,
                     fill=color, outline="")


def icon_pause(c, cx, cy, s, color):
    w, h, gap = s * 0.20, s * 0.90, s * 0.16
    for dx in (-gap - w, gap):
        c.create_rectangle(cx + dx, cy - h / 2.0, cx + dx + w, cy + h / 2.0,
                           fill=color, outline="")


def icon_stop(c, cx, cy, s, color):
    r = s * 0.34
    c.create_rectangle(cx - r, cy - r, cx + r, cy + r, fill=color, outline="")


def icon_volume(c, cx, cy, s, color):
    """喇叭图标（含两道声波弧）。"""
    c.create_polygon(cx - s * 0.70, cy - s * 0.20,
                     cx - s * 0.38, cy - s * 0.20,
                     cx - s * 0.05, cy - s * 0.52,
                     cx - s * 0.05, cy + s * 0.52,
                     cx - s * 0.38, cy + s * 0.20,
                     cx - s * 0.70, cy + s * 0.20,
                     fill=color, outline="")
    for k in (0.42, 0.68):
        r = s * k
        c.create_arc(cx - s * 0.15 - r, cy - r, cx - s * 0.15 + r, cy + r,
                     start=-52, extent=104, style="arc",
                     outline=color, width=1.5)


ICONS = {"play": icon_play, "pause": icon_pause, "stop": icon_stop}


# ---------------------------------------------------------------- 自绘控件

class FlatButton(tk.Canvas):
    """扁平按钮：支持圆形 / 胶囊形，带悬停与按下反馈。"""

    def __init__(self, master, width, height, command=None, shape="circle",
                 icon=None, text=None, bg=C_BG, fill=C_BTN, fill_hover=C_BTN_H,
                 fill_active=C_BTN_A, fg=C_TEXT, fill_disabled="#20242b",
                 fg_disabled="#5b6270", font=None):
        tk.Canvas.__init__(self, master, width=width, height=height, bg=bg,
                           highlightthickness=0, bd=0)
        self._bw, self._bh = width, height
        self._command = command
        self._shape = shape
        self._icon = icon
        self._text = text
        self._fill = fill
        self._fill_hover = fill_hover
        self._fill_active = fill_active
        self._fg = fg
        self._fill_disabled = fill_disabled
        self._fg_disabled = fg_disabled
        self._font = font or FONT_UI
        self._enabled = True
        self._hover = False
        self._pressed = False
        self._redraw()
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<ButtonRelease-1>", self._on_release)

    # ---- 绘制
    def _colors(self):
        if not self._enabled:
            return self._fill_disabled, self._fg_disabled
        if self._pressed:
            return self._fill_active, self._fg
        if self._hover:
            return self._fill_hover, self._fg
        return self._fill, self._fg

    def _redraw(self):
        self.delete("all")
        fill, fg = self._colors()
        if self._shape == "circle":
            self.create_oval(0, 0, self._bw, self._bh, fill=fill, outline="")
        else:
            round_rect(self, 0.5, 0.5, self._bw - 0.5, self._bh - 0.5,
                       (self._bh - 1) / 2.0, fill=fill, outline="")
        cx, cy = self._bw / 2.0, self._bh / 2.0
        if self._icon:
            ICONS[self._icon](self, cx, cy, min(self._bw, self._bh) * 0.5, fg)
        elif self._text:
            self.create_text(cx, cy, text=self._text, fill=fg, font=self._font)

    # ---- 状态
    def set_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled != self._enabled:
            self._enabled = enabled
            self._hover = False
            self._pressed = False
            self._redraw()

    def set_icon(self, icon):
        if icon != self._icon:
            self._icon = icon
            self._redraw()

    # ---- 事件
    def _on_enter(self, _e):
        if self._enabled:
            self._hover = True
            self.configure(cursor="hand2")
            self._redraw()

    def _on_leave(self, _e):
        if self._hover or self._pressed:
            self._hover = False
            self._pressed = False
            self._redraw()

    def _on_press(self, _e):
        if self._enabled:
            self._pressed = True
            self._redraw()

    def _on_release(self, e):
        if not self._enabled:
            return
        was = self._pressed
        self._pressed = False
        inside = (0 <= e.x < self._bw and 0 <= e.y < self._bh)
        self._hover = inside
        self._redraw()
        if was and inside and self._command:
            self._command()


class Bar(tk.Canvas):
    """自绘条形控件（进度条 / 音量条）：可点击定位、可拖动，带悬停高亮。"""

    def __init__(self, master, width=None, height=22, track_h=5, thumb_r=6.5,
                 pad=8, command=None, preview=None, bg=C_PANEL, track=C_TRACK,
                 track_hover=C_TRACK_HL, accent=C_ACCENT, dim=C_TRACK_HL,
                 thumb=C_TEXT):
        kw = {"height": height}
        if width:
            kw["width"] = width
        tk.Canvas.__init__(self, master, bg=bg, highlightthickness=0, bd=0, **kw)
        self.command = command      # 松手后提交 (value: 0.0 ~ 1.0)
        self.preview = preview      # 拖动过程中回调 (value: 0.0 ~ 1.0)
        self.value = 0.0
        self.dragging = False
        self._enabled = True
        self._hover = False
        self._height = height
        self._track_h = track_h
        self._thumb_r = thumb_r
        self._pad = pad
        self._track = track
        self._track_hover = track_hover
        self._accent = accent
        self._dim = dim
        self._thumb = thumb
        self.bind("<Configure>", lambda _e: self._redraw())
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<ButtonPress-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_motion)
        self.bind("<ButtonRelease-1>", self._on_release)

    # ---- 几何
    def _span(self):
        w = self.winfo_width()
        return self._pad + self._thumb_r, w - self._pad - self._thumb_r

    def _frac_from_x(self, x):
        x0, x1 = self._span()
        if x1 <= x0:
            return 0.0
        return max(0.0, min(1.0, (x - x0) / float(x1 - x0)))

    # ---- 绘制
    def _redraw(self):
        self.delete("all")
        if self.winfo_width() <= 1:
            return
        cy = self._height / 2.0
        x0, x1 = self._span()
        if x1 <= x0:
            x1 = x0 + 1
        track = self._track_hover if (self._hover or self.dragging) else self._track
        th = self._track_h
        round_rect(self, x0, cy - th / 2.0, x1, cy + th / 2.0, th / 2.0,
                   fill=track, outline="")
        px = x0 + (x1 - x0) * self.value
        if self.value > 0.001:
            round_rect(self, x0, cy - th / 2.0, max(px, x0 + th), cy + th / 2.0,
                       th / 2.0,
                       fill=self._accent if self._enabled else self._dim,
                       outline="")
        r = self._thumb_r + (1.5 if (self._hover or self.dragging) else 0.0)
        self.create_oval(px - r, cy - r, px + r, cy + r,
                         fill=self._thumb if self._enabled else self._dim,
                         outline="")

    # ---- 状态
    def set_value(self, v):
        if self.dragging:
            return
        v = max(0.0, min(1.0, float(v)))
        if abs(v - self.value) > 0.0002:
            self.value = v
            self._redraw()

    def set_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled != self._enabled:
            self._enabled = enabled
            self._redraw()

    # ---- 事件
    def _on_enter(self, _e):
        self._hover = True
        if self._enabled:
            self.configure(cursor="hand2")
        self._redraw()

    def _on_leave(self, _e):
        self._hover = False
        self._redraw()

    def _on_press(self, e):
        if not self._enabled:
            return
        self.dragging = True
        self.value = self._frac_from_x(e.x)
        self._redraw()
        if self.preview:
            self.preview(self.value)

    def _on_motion(self, e):
        if not self.dragging:
            return
        self.value = self._frac_from_x(e.x)
        self._redraw()
        if self.preview:
            self.preview(self.value)

    def _on_release(self, e):
        if not self.dragging:
            return
        self.dragging = False
        self.value = self._frac_from_x(e.x)
        self._redraw()
        if self.command:
            self.command(self.value)


# ---------------------------------------------------------------- 主程序

class PlayerApp(object):
    """播放器主界面与播放逻辑。"""

    def __init__(self, root):
        self.root = root
        self.player = None          # ffpyplayer.player.MediaPlayer
        self.path = None            # 当前媒体文件路径
        self.state = "stopped"      # stopped / playing / paused
        self.duration = None        # 总时长（秒）
        self.ended = False          # 是否已播放到结尾
        self.has_video = False      # 是否含视频流
        self.src_size = (0, 0)      # 视频源分辨率
        self.out_size = (0, 0)      # 当前输出帧分辨率
        self.volume = 0.8

        self._after_id = None       # 播放循环的定时器
        self._resize_job = None     # 窗口尺寸变化的去抖定时器
        self._photo = None          # 必须持有引用，否则图像会被回收
        self._img_id = None
        self._frame_pts = None      # 最近一次显示画面的时间戳
        self._seek_target = None    # 定位目标（秒）
        self._seek_deadline = 0.0
        self._scrubbing = False     # 暂停状态下为刷新画面而临时恢复解码
        self._autoplay = True
        self._ready = False         # 文件是否已完成打开（ffmpeg 内部 ic 有效）
        self._opened_at = 0.0       # 本次装载的开始时刻
        self._audio_hint_done = False

        self._build_ui()

    # ---------------------------------------------------------------- 界面

    def _build_ui(self):
        self.root.title(APP_TITLE)
        self.root.geometry("920x580")
        self.root.minsize(560, 380)
        self.root.configure(bg=C_BG)

        # 先排底部控制栏，再让画面区占满剩余空间
        panel = tk.Frame(self.root, bg=C_PANEL)
        panel.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Frame(self.root, bg=C_SEP, height=1).pack(side=tk.BOTTOM, fill=tk.X)

        inner = tk.Frame(panel, bg=C_PANEL)
        inner.pack(fill=tk.X, padx=18, pady=(12, 14))

        # 进度条
        self.seek = Bar(inner, height=22, track_h=5, thumb_r=6.5, pad=2,
                        bg=C_PANEL, command=self._on_seek_commit,
                        preview=self._on_seek_preview)
        self.seek.pack(fill=tk.X)

        # 控制行
        row = tk.Frame(inner, bg=C_PANEL)
        row.pack(fill=tk.X, pady=(10, 0))

        # --- 左侧按钮
        self.btn_open = FlatButton(row, 86, 34, shape="pill", text="打开文件",
                                   bg=C_PANEL, fill=C_BTN, fill_hover=C_BTN_H,
                                   fill_active=C_BTN_A, fg=C_TEXT,
                                   font=FONT_UI, command=self.open_file)
        self.btn_open.pack(side=tk.LEFT)

        self.btn_play = FlatButton(row, 44, 44, shape="circle", icon="play",
                                   bg=C_PANEL, fill=C_ACCENT,
                                   fill_hover=C_ACCENT_H, fill_active=C_ACCENT_A,
                                   fg="#ffffff", fill_disabled="#232830",
                                   fg_disabled="#5b6270",
                                   command=self.toggle_play)
        self.btn_play.pack(side=tk.LEFT, padx=(10, 0))

        self.btn_stop = FlatButton(row, 34, 34, shape="circle", icon="stop",
                                   bg=C_PANEL, fill=C_BTN, fill_hover=C_BTN_H,
                                   fill_active=C_BTN_A, fg=C_TEXT,
                                   fill_disabled="#232830", fg_disabled="#5b6270",
                                   command=self.stop)
        self.btn_stop.pack(side=tk.LEFT, padx=(8, 0))

        # --- 右侧：时间 + 音量（side=RIGHT 时后 pack 的排在更左边）
        self.vol = Bar(row, width=104, height=20, track_h=4, thumb_r=5.5, pad=2,
                       bg=C_PANEL, command=self._on_volume)
        self.vol.pack(side=tk.RIGHT, padx=(0, 2))
        self.vol.set_value(self.volume)

        vol_icon = tk.Canvas(row, width=24, height=24, bg=C_PANEL,
                             highlightthickness=0, bd=0)
        vol_icon.pack(side=tk.RIGHT, padx=(6, 0))
        icon_volume(vol_icon, 12, 12, 22, C_TEXT_DIM)

        self.lbl_total = tk.Label(row, text="--:--", bg=C_PANEL, fg=C_TEXT_DIM,
                                  font=FONT_TIME)
        self.lbl_total.pack(side=tk.RIGHT, padx=(4, 0))
        tk.Label(row, text="/", bg=C_PANEL, fg="#4c5361",
                 font=FONT_TIME).pack(side=tk.RIGHT, padx=2)
        self.lbl_pos = tk.Label(row, text="--:--", bg=C_PANEL, fg=C_TEXT,
                                font=FONT_TIME)
        self.lbl_pos.pack(side=tk.RIGHT)

        # --- 画面区
        self.canvas = tk.Canvas(self.root, bg="#000000",
                                highlightthickness=0, bd=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self._img_id = self.canvas.create_image(0, 0, anchor="center")
        self._hint_id = self.canvas.create_text(
            0, 0, anchor="center", fill=C_HINT, font=FONT_HINT,
            justify="center",
            text="把音视频文件拖到这里，或点击左下角「打开文件」\n"
                 "支持 mp4 / mkv / avi / mov / mp3 / flac / wav 等常见格式")
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._refresh_controls()
        self._setup_dnd()

    # ---------------------------------------------------------------- 拖放

    def _setup_dnd(self):
        """把窗口内所有控件都注册成文件拖放目标，整窗任意位置都能接。"""
        if not HAS_DND:
            return

        def walk(widget):
            try:
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<DropEnter>>", self._on_drop_enter)
                widget.dnd_bind("<<DropPosition>>", self._on_drop_enter)
                widget.dnd_bind("<<Drop>>", self._on_drop)
            except Exception:
                pass
            for child in widget.winfo_children():
                walk(child)

        walk(self.root)

    @staticmethod
    def _dnd_action(event, default="copy"):
        """取出并校验拖放动作。

        tkinterdnd2 没有用 tkinter 标准的 if/break 包装，回调的返回值会**直接
        交给 tkdnd**，所以这些回调必须返回一个合法动作（copy/move/link/...），
        返回 None 会让这次拖放被判为无效。
        """
        action = str(getattr(event, "action", "") or "").lower()
        if action not in ("copy", "move", "link", "ask", "private"):
            action = default
        return action

    def _on_drop_enter(self, event):
        """拖到窗口上方时返回接受动作，否则系统会显示禁止光标。"""
        try:
            self.root.configure(cursor="hand2")
        except Exception:
            pass
        return self._dnd_action(event)

    def _on_drop(self, event):
        """接收拖进来的文件（Tk 会用 {} 把带空格的路径包起来，交给 splitlist 拆）。"""
        try:
            items = self.root.tk.splitlist(event.data)
        except Exception:
            items = [event.data]
        for item in items:
            path = os.path.normpath(str(item))
            if os.path.isfile(path):
                self.load(path, autoplay=True)
                break
        try:
            self.root.configure(cursor="")
        except Exception:
            pass
        return self._dnd_action(event)

    def _set_hint(self, text):
        """在画面上显示提示文字（无视频时用）。"""
        try:
            # 先摘掉画面项上的图片，避免它仍指向已销毁的 PhotoImage
            self.canvas.itemconfigure(self._img_id, image="", state="hidden")
        except tk.TclError:
            pass
        self.canvas.itemconfig(self._hint_id, text=text, state="normal")

    def _show_video(self):
        self.canvas.itemconfig(self._hint_id, state="hidden")
        self.canvas.itemconfig(self._img_id, state="normal")

    def _set_time(self, pos, total):
        self.lbl_pos.config(text=fmt_time(pos))
        self.lbl_total.config(text=fmt_time(total) if total else "--:--")

    def _refresh_controls(self):
        self.btn_play.set_icon("pause" if self.state == "playing" else "play")
        has_media = (self.player is not None) or (self.path is not None)
        self.btn_play.set_enabled(has_media)
        self.btn_stop.set_enabled(self.state != "stopped")
        self.seek.set_enabled(self.duration is not None)

    # ------------------------------------------------------------ 打开/装载

    def open_file(self):
        path = filedialog.askopenfilename(
            title="选择要播放的音乐或视频",
            filetypes=[
                ("所有支持的媒体", " ".join("*." + e for e in VIDEO_EXTS + AUDIO_EXTS)),
                ("视频文件", " ".join("*." + e for e in VIDEO_EXTS)),
                ("音频文件", " ".join("*." + e for e in AUDIO_EXTS)),
                ("所有文件", "*.*"),
            ])
        if path:
            self.load(path, autoplay=True)

    def load(self, path, autoplay=True):
        """装载媒体文件并（可选）立即播放。"""
        self.stop()
        if not os.path.isfile(path):
            messagebox.showerror(APP_TITLE, "文件不存在：\n%s" % path)
            return
        self.path = path
        self.root.title("%s - %s" % (os.path.basename(path), APP_TITLE))
        self._set_hint("正在打开……\n%s" % os.path.basename(path))
        self.root.update_idletasks()

        try:
            self.player = MediaPlayer(
                path,
                ff_opts={
                    "paused": True,       # 先暂停，等元数据就绪再决定是否自动播放
                    "out_fmt": "rgb24",   # 交给 Tk 显示的像素格式
                    "sync": "audio",      # 以音频为主时钟，保证音画同步
                    "volume": self.volume,
                })
        except Exception as exc:
            self.player = None
            self._set_hint("无法打开该文件\n%s" % exc)
            messagebox.showerror(APP_TITLE, "无法打开文件：\n%s\n\n%s" % (path, exc))
            return

        self.state = "paused"
        self.ended = False
        self.duration = None
        self.has_video = False
        self.src_size = (0, 0)
        self.out_size = (0, 0)
        self._seek_target = None
        self._scrubbing = False
        self._ready = False
        self._opened_at = time.time()
        self._audio_hint_done = False
        self._frame_pts = None
        self._refresh_controls()

        self._autoplay = autoplay
        self._tick()

    def _sync_metadata(self):
        """同步「时长」和「是否含视频流」。

        ffpyplayer 的 src_vid_size 是视频线程解出第一帧之后才填的，比 duration
        晚（大文件实测差 100ms 以上）。所以不能一拿到 duration 就断定这是纯音频
        —— 那会把视频文件锁死成音频。这里每次都重新判，允许之后纠正过来。
        """
        md = self.player.get_metadata()

        duration = md.get("duration")
        if duration is not None and not self._ready:
            self._ready = True
            self.duration = duration or None
            self._set_time(0, self.duration)
            if self._autoplay:
                self.player.set_pause(False)
                self.state = "playing"
            self._refresh_controls()

        src = md.get("src_vid_size") or (0, 0)
        if src[0] and src[1]:
            if not self.has_video:
                # 有视频流：可能上一轮已经误判成音频，这里纠正回来
                self.has_video = True
                self.src_size = (int(src[0]), int(src[1]))
                self._show_video()
                self._fit_size(force=True)
                self._refresh_controls()
        elif self._ready and not self._audio_hint_done and \
                time.time() - self._opened_at >= AUDIO_PROBE_DELAY:
            # 过了观察期还是没有视频流信息，才认定是纯音频
            self._audio_hint_done = True
            self._set_hint("♪ 正在播放音频\n%s" % os.path.basename(self.path or ""))

    # ------------------------------------------------------------ 播放控制

    def toggle_play(self):
        if self.state == "playing":
            self.player.set_pause(True)
            self.state = "paused"
        elif self.state == "paused":
            if self.ended:
                # 播完后再次播放：从头重新装载
                self.load(self.path, autoplay=True)
                return
            self.player.set_pause(False)
            self.state = "playing"
        else:  # stopped，重新装载并播放
            if self.path:
                self.load(self.path, autoplay=True)
        self._refresh_controls()

    def stop(self):
        """停止播放并释放解码器。"""
        self._cancel_tick()
        self._close_player()
        self.player = None
        self.state = "stopped"
        self.ended = False
        self.duration = None
        self.has_video = False
        self.src_size = (0, 0)
        self._seek_target = None
        self._scrubbing = False
        self._ready = False
        self._frame_pts = None
        self.seek.set_value(0)
        self._set_time(None, None)
        if self.path:
            self._set_hint("已停止\n%s" % os.path.basename(self.path))
        else:
            self._set_hint("把音视频文件拖到这里，或点击左下角「打开文件」\n"
                           "支持 mp4 / mkv / avi / mov / mp3 / flac / wav 等常见格式")
        self._photo = None
        self._refresh_controls()

    def on_close(self):
        self._cancel_tick()
        self._close_player()
        self.root.destroy()

    def _close_player(self):
        if self.player is not None:
            try:
                self.player.close_player()
            except Exception:
                pass
        self.player = None

    # ------------------------------------------------------------ 音量/定位

    def _on_volume(self, value):
        self.volume = max(0.0, min(1.0, float(value)))
        if self.player is not None:
            self.player.set_volume(self.volume)

    def _on_seek_preview(self, frac):
        # 拖动过程中只更新时间显示，松手才真正定位
        if self.duration:
            self._set_time(frac * self.duration, self.duration)

    def _on_seek_commit(self, frac):
        if self.player is None or not self.duration:
            return
        target = max(0.0, min(frac * self.duration, self.duration - 0.1))
        try:
            self.player.seek(target, relative=False)
        except Exception:
            return
        if self.has_video:
            # 定位后解码器里可能还有旧帧，先丢弃直到接近目标位置。
            # 纯音频没有画面要刷新，不需要这一步。
            self._seek_target = target
            self._seek_deadline = time.time() + SEEK_TIMEOUT

    # ------------------------------------------------------------ 播放循环

    def _cancel_tick(self):
        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except Exception:
                pass
            self._after_id = None

    def _schedule(self, delay):
        self._cancel_tick()
        delay = 0.001 if delay is None else max(0.001, float(delay))
        self._after_id = self.root.after(max(1, int(delay * 1000)), self._tick)

    def _tick(self):
        """播放循环：取帧 -> 显示 -> 刷新进度 -> 按解码器节奏安排下一次。"""
        self._after_id = None
        player = self.player
        if player is None:
            return

        # 反复同步元数据：src_vid_size 比 duration 晚到，所以只要还没确认有
        # 视频流就继续看，避免把视频文件误判成音频。
        # 注意：文件真正打开（duration 被填充）之前不能调用 get_pts()，
        # 否则 ffmpeg 内部空指针会导致整个进程崩溃，所以 _update_progress 有 _ready 门控。
        if not self._ready or not self.has_video:
            self._sync_metadata()

        frame, val = player.get_frame()

        if val == "eof":
            self._on_eof()
            return
        if val == "paused":
            self._update_progress()
            if self._scrubbing:
                if self._seek_target is None:
                    # 画面已刷新到目标位置（或超时），恢复暂停
                    player.set_pause(True)
                    self._scrubbing = False
                    self._schedule(0.1)
                else:
                    self._schedule(0.01)
                return
            if self._seek_target is not None and self.state == "paused":
                # 暂停时定位：临时放开解码，取到目标位置的画面后再暂停回去
                player.set_pause(False)
                self._scrubbing = True
                self._schedule(0.01)
                return
            self._schedule(0.1)
            return

        # 定位后把「还没跳到目标位置」的旧帧一次丢完。
        # 必须批量丢：Tk 定时器精度约 15ms，一帧挂一个 after 的话，
        # 队列里积压几十帧就要好几秒，会超过 SEEK_TIMEOUT 而停在错误的帧上。
        dropped = 0
        while frame is not None and dropped < 500 and self._drop_frame(frame):
            dropped += 1
            frame, val = player.get_frame()
            if val == "eof":
                self._on_eof()
                return

        if frame is not None:
            self._render_frame(frame)

        self._update_progress()
        if self._scrubbing and self._seek_target is None:
            player.set_pause(True)
            self._scrubbing = False
        self._schedule(val if isinstance(val, (int, float)) else 0.01)

    def _drop_frame(self, frame):
        """定位后，丢弃尚未跳到目标位置的旧帧。"""
        if self._seek_target is None:
            return False
        pts = frame[1]
        if abs(pts - self._seek_target) <= SEEK_TOLERANCE or \
                time.time() >= self._seek_deadline:
            self._seek_target = None
            return False
        return True

    def _current_pos(self):
        """当前播放位置（秒）。

        有视频时以「屏幕上已显示那一帧」的时间戳为准，保证进度条和时间显示
        一定和画面一致。定位后 ffplay 需要几秒把画面拉回音频主时钟，这期间
        音频时钟会明显落后于画面，直接用它会显示对不上。
        画面已放完但音频还在播（罕见）时回退到时钟，避免进度卡住。
        """
        try:
            clk = self.player.get_pts()
        except Exception:
            clk = None
        if clk is not None and (clk < 0 or clk != clk):     # 负值或 NaN
            clk = None
        if self.has_video and self._frame_pts is not None:
            if clk is None or clk <= self._frame_pts + 1.0:
                return self._frame_pts
        return clk if clk is not None else 0.0

    def _update_progress(self):
        if not self._ready or self.seek.dragging:
            return
        if self._seek_target is not None and time.time() >= self._seek_deadline:
            self._seek_target = None

        total = self.duration
        pos = max(0.0, self._current_pos())
        if total:
            pos = min(pos, total)
            self.seek.set_value(pos / total)
        self._set_time(pos, total)

    def _on_eof(self):
        self.ended = True
        self.state = "paused"
        self._scrubbing = False
        self._seek_target = None
        if self.duration:
            self.seek.set_value(1.0)
            self._set_time(self.duration, self.duration)
        self._refresh_controls()

    # ------------------------------------------------------------ 视频渲染

    def _on_canvas_configure(self, event):
        self.canvas.coords(self._img_id, event.width // 2, event.height // 2)
        self.canvas.coords(self._hint_id, event.width // 2, event.height // 2)
        if not self.has_video:
            return
        if self._resize_job is not None:
            try:
                self.root.after_cancel(self._resize_job)
            except Exception:
                pass
        self._resize_job = self.root.after(150, self._fit_size)

    def _fit_size(self, force=False):
        """按窗口大小等比缩放输出帧（由 FFmpeg 的 swscale 完成，效率高）。"""
        if self.player is None or not self.has_video:
            return
        self._resize_job = None
        sw, sh = self.src_size
        cw = max(2, self.canvas.winfo_width())
        ch = max(2, self.canvas.winfo_height())
        if sw <= 0 or sh <= 0:
            return
        k = min(cw / float(sw), ch / float(sh))
        tw = max(2, int(sw * k)) & ~1     # 取偶数，避免部分色度格式报错
        th = max(2, int(sh * k)) & ~1
        if force or (tw, th) != self.out_size:
            self.out_size = (tw, th)
            try:
                self.player.set_size(tw, th)
            except Exception:
                pass

    def _render_frame(self, frame):
        img, pts = frame
        w, h = img.get_size()
        data = bytes(img.to_bytearray()[0])
        if len(data) < w * h * 3:
            return
        # Tk 8.6 原生支持 PPM(P6)，无需额外依赖即可显示原始 RGB 数据
        ppm = b"P6\n%d %d\n255\n" % (w, h) + data
        try:
            photo = tk.PhotoImage(master=self.canvas, data=ppm)
        except tk.TclError:
            return
        self._photo = photo          # 保持引用
        if isinstance(pts, float) and pts == pts:      # 排除 NaN
            self._frame_pts = pts
        self.canvas.itemconfig(self._img_id, image=photo)


def main():
    root = TkinterDnD.Tk() if HAS_DND else tk.Tk()
    app = PlayerApp(root)
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        root.after(60, lambda: app.load(sys.argv[1], autoplay=True))
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # 双击运行时能看见错误
        import traceback
        traceback.print_exc()
        try:
            r = tk.Tk()
            r.withdraw()
            messagebox.showerror(APP_TITLE, "程序异常退出：\n%s" % exc)
            r.destroy()
        except Exception:
            pass
        sys.exit(1)
