"""PCM WAV sampling and a Tk Canvas view; Python standard library only."""
from array import array
import math
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import sys
import wave


def decode_pcm(data, width):
    """Normalize little-endian integer PCM to [-1, 1), including unsigned 8-bit."""
    if width == 1:
        return [(value - 128) / 128.0 for value in data]
    if width in (2, 4):
        values = array('h' if width == 2 else 'i')
        values.frombytes(data)
        if sys.byteorder != 'little':
            values.byteswap()
        scale = float(1 << (width * 8 - 1))
        return [value / scale for value in values]
    if width == 3:
        return [int.from_bytes(data[i:i + 3], 'little', signed=True) / 8388608.0
                for i in range(0, len(data), 3)]
    raise ValueError('仅支持 8 / 16 / 24 / 32 位整数 PCM WAV')


def dbfs(value):
    return 20.0 * math.log10(value) if value > 0 else float('-inf')


class WavReader:
    """Read a bounded 60 ms window on demand; never load a whole track into RAM."""
    def __init__(self, path):
        self._file = wave.open(str(path), 'rb')
        try:
            self.channels = self._file.getnchannels()
            self.rate = self._file.getframerate()
            self.width = self._file.getsampwidth()
            self.frames = self._file.getnframes()
            if (self._file.getcomptype() != 'NONE' or self.width not in (1, 2, 3, 4)
                    or not 1 <= self.channels <= 8 or self.rate <= 0 or self.frames <= 0):
                raise ValueError('不支持的 WAV 采样格式')
            self.duration = self.frames / self.rate
        except BaseException:
            self._file.close()
            raise

    def close(self):
        self._file.close()

    def snapshot(self, position, points=240):
        position = max(0.0, min(float(position), self.duration))
        start = min(self.frames, int(position * self.rate))
        count = min(8192, max(64, int(self.rate * 0.06)), self.frames - start)
        self._file.setpos(start)
        raw = self._file.readframes(count)
        frame_bytes = self.width * self.channels
        raw = raw[:len(raw) // frame_bytes * frame_bytes]
        samples = decode_pcm(raw, self.width)
        channels = []
        # Keep stereo channels separate: opposite-phase signals must not cancel.
        for channel in range(min(self.channels, 2)):
            values = samples[channel::self.channels]
            total = len(values)
            buckets = max(1, min(int(points), total or 1))
            envelope = []
            for i in range(buckets):
                section = values[i * total // buckets:(i + 1) * total // buckets]
                envelope.append((min(section), max(section)) if section else (0.0, 0.0))
            rms = math.sqrt(sum(value * value for value in values) / total) if total else 0.0
            channels.append({'envelope': envelope, 'rms': rms,
                             'peak': max((abs(value) for value in values), default=0.0)})
        return {'position': position, 'window': count / self.rate, 'channels': channels}



def find_ffmpeg():
    """Use the executable already distributed with ffpyplayer, never download one."""
    import ffpyplayer
    prefixes = (Path(sys.prefix), Path(ffpyplayer.__file__).resolve().parents[3])
    for prefix in prefixes:
        binary = prefix / 'share' / 'ffpyplayer' / 'ffmpeg' / 'bin' / ('ffmpeg.exe' if os.name == 'nt' else 'ffmpeg')
        if binary.is_file():
            return binary
    raise OSError('当前 ffpyplayer 安装中未找到自带的 FFmpeg')


class SpectrumReader:
    """Bounded, asynchronous frequency analysis using the existing FFmpeg filter.

    Audio playback stays in MediaPlayer. The analysis subprocess has no audio output.
    Queue timestamps follow the playback clock; seeking restarts only this decoder.
    """
    BANDS = 48
    WIDTH = 96
    HEIGHT = 128
    FPS = 25

    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.binary = find_ffmpeg()
        self.channels = None
        try:
            probe = WavReader(path)
            self.channels = probe.channels
            probe.close()
        except (OSError, EOFError, ValueError, wave.Error):
            pass
        self._process = None
        self._thread = None
        self._stop = None
        self._queue = None
        self._next = None
        self._last = None
        self._last_requested = None
        self._seek_request = None
        self._last_restart = 0.0
        self._levels = [0.0] * self.BANDS
        self.error = None

    @property
    def pending(self):
        return self._last is None or self._seek_request is not None

    @classmethod
    def _levels_from_frame(cls, raw):
        levels = []
        for band in range(cls.BANDS):
            x = band * 2
            first = cls.HEIGHT - 1
            for y in range(cls.HEIGHT):
                offset = (y * cls.WIDTH + x) * 3
                if raw[offset] > 32 or raw[offset + 3] > 32:
                    first = y
                    break
            levels.append((cls.HEIGHT - 1 - first) / (cls.HEIGHT - 1))
        return levels

    def _start(self, position):
        self._stop_process()
        self._queue = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._last = self._next = None
        self._levels = [0.0] * self.BANDS
        self.error = None
        self._last_restart = time.monotonic()
        colors = '|'.join(['white'] * 8)
        filter_graph = (
            '[0:a:0]asetpts=PTS-STARTPTS,'
            'showfreqs=s=96x128:rate=25:mode=bar:ascale=log:fscale=log:'
            'win_size=2048:win_func=hann:overlap=0.5:averaging=1:colors=' + colors
            + ',format=rgb24[v]')
        command = [str(self.binary), '-hide_banner', '-loglevel', 'error', '-nostdin',
                   '-ss', '%.6f' % position, '-i', self.path, '-filter_complex', filter_graph,
                   '-map', '[v]', '-an', '-sn', '-dn', '-pix_fmt', 'rgb24',
                   '-f', 'rawvideo', 'pipe:1']
        self._process = process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            bufsize=self.WIDTH * self.HEIGHT * 3 * 2,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        frame_queue, stop = self._queue, self._stop

        def consume():
            frame_size = self.WIDTH * self.HEIGHT * 3
            index = 0
            try:
                while not stop.is_set():
                    raw = process.stdout.read(frame_size)
                    if len(raw) != frame_size:
                        break
                    item = (position + index / self.FPS, self._levels_from_frame(raw))
                    index += 1
                    while not stop.is_set():
                        try:
                            frame_queue.put(item, timeout=.05)
                            break
                        except queue.Full:
                            continue
                if not stop.is_set():
                    code = process.wait()
                    if code and self._process is process:
                        self.error = '无法分析此文件的音频频谱'
            except (OSError, ValueError):
                if not stop.is_set() and self._process is process:
                    self.error = '频谱分析已停止'

        self._thread = threading.Thread(target=consume, daemon=True)
        self._thread.start()

    def _stop_process(self):
        process, worker = self._process, self._thread
        if self._stop is not None:
            self._stop.set()
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=.4)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=.4)
            if worker is not None:
                worker.join(timeout=.2)
            if process.stdout is not None:
                process.stdout.close()
        self._process = self._thread = None

    def close(self):
        self._stop_process()
        self._last = self._next = None
        self._last_requested = None
        self._seek_request = None
        self._levels = [0.0] * self.BANDS

    def snapshot(self, position, points=None):
        position = max(0.0, float(position))
        if self.error:
            raise OSError(self.error)
        jump = self._last_requested is not None and (
            position < self._last_requested - .08 or position > self._last_requested + .35)
        if self._process is None or jump:
            self._seek_request = position
        self._last_requested = position
        if self._seek_request is not None:
            self._seek_request = position
            if self._process is None or time.monotonic() - self._last_restart >= .12:
                self._start(position)
                self._seek_request = None
            else:
                return {'position': position, 'levels': self._levels[:]}
        for _ in range(9):
            if self._next is None:
                try:
                    self._next = self._queue.get_nowait()
                except queue.Empty:
                    break
            if self._next[0] > position + .04:
                break
            self._last = self._next
            self._levels = self._next[1]
            self._next = None
        return {'position': position, 'levels': self._levels[:]}


class WavView:
    """A restrained decorative spectrum: low frequencies left, high frequencies right."""
    TAG = 'wav_visual'

    def __init__(self, canvas):
        self.canvas = canvas
        self._levels = []
        self._peaks = []

    def clear(self):
        self.canvas.delete(self.TAG)
        self._levels = []
        self._peaks = []

    @staticmethod
    def display_level(value):
        # showfreqs compresses roughly 120 dB into [0, 1]. Expand the musically
        # useful part so a beat changes height instead of leaving every bar high.
        value = max(0.0, min(1.0, (value - .25) / .75))
        return value ** 2.25

    @staticmethod
    def _blend(a, b, amount):
        left = tuple(int(a[i:i + 2], 16) for i in (1, 3, 5))
        right = tuple(int(b[i:i + 2], 16) for i in (1, 3, 5))
        return '#' + ''.join('%02x' % round(x + (y - x) * amount) for x, y in zip(left, right))

    def draw(self, snapshot, reset=False, paused=False):
        c = self.canvas
        c.delete(self.TAG)
        width, height = max(2, c.winfo_width()), max(2, c.winfo_height())
        if width < 200 or height < 160:
            return
        background = '#10131c'
        c.create_rectangle(0, 0, width, height, fill=background, outline='', tags=self.TAG)
        incoming = snapshot['levels']
        # A light frequency-domain blur softens gaps without moving bass to treble.
        targets = []
        for i, value in enumerate(incoming):
            neighbors = (incoming[max(0, i - 1)] + incoming[min(len(incoming) - 1, i + 1)]) / 2
            targets.append(self.display_level(max(value, neighbors * .30)))
        if reset or paused or len(self._levels) != len(targets):
            self._levels = targets[:]
            self._peaks = targets[:]
        elif not paused:
            for i, target in enumerate(targets):
                strength = .94 if target > self._levels[i] else .50
                self._levels[i] += (target - self._levels[i]) * strength
                self._peaks[i] = max(self._levels[i], self._peaks[i] - .05)
        left, right = width * .075, width * .925
        baseline = height * .68
        amplitude = height * .54
        stride = (right - left) / len(targets)
        bar_width = max(2, stride * .58)
        c.create_line(left, baseline + 2, right, baseline + 2,
                      fill='#28303e', width=1, tags=self.TAG)
        for i, value in enumerate(self._levels):
            ratio = i / max(1, len(targets) - 1)
            color = (self._blend('#52dbc6', '#6fa4ff', ratio * 2) if ratio <= .5
                     else self._blend('#6fa4ff', '#c789f4', (ratio - .5) * 2))
            x = left + stride * i + (stride - bar_width) / 2
            bar_height = max(2.0, value * amplitude)
            # Six stacked shades create a soft gradient without an image/UI library.
            for step in range(6):
                y0 = baseline - bar_height * (step + 1) / 6
                y1 = baseline - bar_height * step / 6
                fill = self._blend(color, '#e9faff', step / 6 * .27)
                c.create_rectangle(x, y0, x + bar_width, y1, fill=fill, outline='', tags=self.TAG)
            if value > .015:
                peak_y = baseline - self._peaks[i] * amplitude - 6
                c.create_rectangle(x, peak_y, x + bar_width, peak_y + 2,
                                   fill=self._blend(color, '#ffffff', .4), outline='', tags=self.TAG)
                for step in range(3):
                    y0 = baseline + 8 + bar_height * .22 * step / 3
                    y1 = baseline + 8 + bar_height * .22 * (step + 1) / 3
                    fill = self._blend(background, color, .16 * (1 - step / 3))
                    c.create_rectangle(x, y0, x + bar_width, y1, fill=fill, outline='', tags=self.TAG)
