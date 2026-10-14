"""Extract completed ZIP members while a resumable archive download is in progress."""

import io
import os
from pathlib import Path
import time
import zipfile


class IncompleteDownload(OSError):
    pass


class DownloadedPrefix(io.RawIOBase):
    def __init__(self, path, tail, size):
        self.stream = open(path, "rb")
        self.tail = Path(tail).read_bytes()
        self.size = size
        self.position = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        if self.position < 0:
            raise ValueError("Negative ZIP offset.")
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        if size < 0:
            size = self.size - self.position
        size = min(size, self.size - self.position)
        if self.position >= self.size - len(self.tail):
            offset = self.position - (self.size - len(self.tail))
            result = self.tail[offset:offset + size]
        elif self.position + size <= os.fstat(self.stream.fileno()).st_size:
            self.stream.seek(self.position)
            result = self.stream.read(size)
        else:
            raise IncompleteDownload("Requested ZIP member has not finished downloading.")
        self.position += len(result)
        return result

    def close(self):
        self.stream.close()
        super().close()


def extract_available(path, out_dir, tail_path=None, total_bytes=None):
    out_dir = Path(out_dir).resolve()
    reader = DownloadedPrefix(path, tail_path, total_bytes) if tail_path else open(path, "rb")
    completed, pending = 0, 0
    try:
        with zipfile.ZipFile(reader) as archive:
            for item in archive.infolist():
                if item.is_dir():
                    continue
                destination = (out_dir / item.filename).resolve()
                if not destination.is_relative_to(out_dir):
                    raise ValueError("Unsafe archive path.")
                if destination.is_file() and destination.stat().st_size == item.file_size:
                    completed += 1
                    continue
                # Account for local header and its possible filename/extra fields.
                prefix_size = Path(path).stat().st_size
                if tail_path and prefix_size < item.header_offset + item.compress_size + 65536:
                    pending += 1
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix(destination.suffix + ".partial")
                try:
                    with archive.open(item) as source, temporary.open("wb") as target:
                        while chunk := source.read(1024 * 1024):
                            started = time.monotonic()
                            target.write(chunk)
                            time.sleep(max(0.001, (time.monotonic() - started) * 3))
                    temporary.replace(destination)
                    print(f"Extracted and CRC-verified {item.filename}", flush=True)
                    completed += 1
                except IncompleteDownload:
                    pending += 1
    finally:
        reader.close()
    return {"completed": completed, "pending": pending}
