"""ffind Web 界面 —— 零依赖 http.server + 单文件前端。

仅监听 127.0.0.1(只本机可访问)。搜索复用 CLI 同一套 Index.query,
重建索引在后台线程跑,前端轮询 /api/status 拿进度。

路由:
    GET  /            单页前端(web/index.html)
    GET  /api/status  索引元信息 + 重建进度
    GET  /api/search  查询(走 SQLite 索引,毫秒级)
    POST /api/update  触发后台重建
    POST /api/open    xdg-open 打开结果(限索引根范围内)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import scan
from .index import Index

_WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
_MAX_BODY = 64 * 1024


def _stat_row(path: str, is_dir: bool) -> dict:
    size = mtime = None
    if not is_dir:
        try:
            st = os.stat(path)
            size, mtime = st.st_size, st.st_mtime
        except OSError:
            pass
    return {
        "path": path,
        "dir": is_dir,
        "name": os.path.basename(path) or path,
        "size": size,
        "mtime": mtime,
    }


def make_server(
    idx: Index,
    roots: list[str],
    host: str = "127.0.0.1",
    port: int = 8765,
) -> ThreadingHTTPServer:
    """构造 HTTP 服务器(不启动)。port=0 时自动选端口,
    用 ``srv.server_address[1]`` 取实际端口 —— 便于测试。"""
    roots = scan.normalize_roots(roots)
    lock = threading.Lock()
    state = {"running": False, "files": 0, "dirs": 0, "seconds": 0.0, "error": None}

    class Handler(BaseHTTPRequestHandler):
        server_version = "ffind/0.1"

        def log_message(self, *args):  # 安静:不打印每个请求
            pass

        # ---------- 发送 ----------
        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                self.wfile.write(body)
            except BrokenPipeError:
                pass

        def _json(self, obj, code: int = 200):
            self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                       "application/json; charset=utf-8")

        def _read_body(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n <= 0:
                return {}
            if n > _MAX_BODY:
                raise ValueError("请求体过大")
            raw = self.rfile.read(n)
            return json.loads(raw.decode("utf-8") or "{}")

        # ---------- GET ----------
        def do_GET(self):
            u = urlparse(self.path)
            try:
                if u.path in ("/", "/index.html"):
                    self._serve_index()
                elif u.path == "/favicon.ico":
                    self._send(204, b"", "image/x-icon")
                elif u.path == "/api/status":
                    self._json(self._status())
                elif u.path == "/api/search":
                    self._json(self._search(u))
                else:
                    self._json({"error": "not found"}, 404)
            except BrokenPipeError:
                pass
            except Exception as e:  # noqa: BLE001 —— 统一转成 JSON 错误
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _serve_index(self):
            path = os.path.join(_WEB_DIR, "index.html")
            try:
                with open(path, "rb") as f:
                    self._send(200, f.read(), "text/html; charset=utf-8")
            except OSError:
                self._send(500, b"web/index.html missing", "text/plain; charset=utf-8")

        def _status(self) -> dict:
            st = idx.stats()
            st["excludes"] = sorted(st["excludes"])
            st["roots"] = roots[:]
            with lock:
                st["rebuild"] = dict(state)
            return st

        def _search(self, u) -> dict:
            q = parse_qs(u.query)
            query = (q.get("q", [""])[0] or "").strip()
            if not query:
                return {"hits": [], "total": 0, "capped": False, "ms": 0.0}
            try:
                n = max(1, min(2000, int(q.get("n", ["300"])[0])))
            except ValueError:
                n = 300
            typ = q.get("type", ["any"])[0]
            if typ not in ("any", "file", "dir"):
                typ = "any"
            frm = [f for f in q.get("from", []) if f]
            exts = [e for e in (q.get("ext", [""])[0].split(",") if q.get("ext") else []) if e.strip()]

            t0 = time.perf_counter()
            hits, total, capped = idx.query(
                query.split(), type_=typ, roots=frm, exts=exts, limit=n
            )
            rows = [_stat_row(p, d) for p, d in hits]
            return {
                "hits": rows,
                "total": total,
                "capped": capped,
                "ms": round((time.perf_counter() - t0) * 1000, 1),
            }

        # ---------- POST ----------
        def do_POST(self):
            u = urlparse(self.path)
            try:
                if u.path == "/api/update":
                    self._update()
                elif u.path == "/api/open":
                    self._open()
                else:
                    self._json({"error": "not found"}, 404)
            except BrokenPipeError:
                pass
            except Exception as e:  # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"}, 500)

        def _update(self):
            data = self._read_body()
            want = scan.normalize_roots(data.get("roots") or roots)
            want = [r for r in want if os.path.isdir(r)]
            if not want:
                return self._json({"error": "没有有效目录"}, 400)
            with lock:
                if state["running"]:
                    return self._json({"error": "正在重建中"}, 409)
                state.update(running=True, files=0, dirs=0, seconds=0.0, error=None)

            last = {"t": 0.0}

            def tick(f: int, d: int, s: float):
                now = time.monotonic()
                if now - last["t"] >= 0.25:
                    last["t"] = now
                    with lock:
                        state.update(files=f, dirs=d, seconds=s)

            def worker():
                try:
                    excl = idx.stored_excludes() | set(data.get("exclude_add") or [])
                    st = idx.build(want, excludes=frozenset(excl), progress=tick)
                    roots[:] = want  # 打开范围随之更新
                    with lock:
                        state.update(running=False, files=st["files"],
                                     dirs=st["dirs"], seconds=st["seconds"])
                except Exception as e:  # noqa: BLE001
                    with lock:
                        state.update(running=False, error=f"{type(e).__name__}: {e}")

            threading.Thread(target=worker, daemon=True).start()
            self._json({"ok": True, "roots": want})

        def _open(self):
            data = self._read_body()
            target = data.get("path", "")
            if not target:
                return self._json({"error": "缺少 path"}, 400)
            real = os.path.realpath(target)
            allowed = any(real == r or real.startswith(r + os.sep) for r in roots)
            if not allowed or not os.path.exists(real):
                return self._json({"error": "路径不在索引根范围内"}, 403)
            cmd = shutil.which("xdg-open") or shutil.which("open")
            if not cmd:
                return self._json({"ok": False, "msg": "系统无 xdg-open,请用复制路径"})
            try:
                subprocess.Popen([cmd, target],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError as e:
                return self._json({"error": str(e)}, 500)
            self._json({"ok": True})

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    return srv


def serve(db=None, port: int = 8765, roots=None, open_browser: bool = False) -> int:
    """启动 Web 界面并阻塞运行,直到 Ctrl-C。"""
    idx = Index(db)
    rs = scan.normalize_roots(roots or []) or idx.stored_roots() or [os.path.expanduser("~")]
    try:
        srv = make_server(idx, rs, port=port)
    except OSError as e:
        print(f"ffind: 端口 {port} 不可用({e});用 --port 换一个。", flush=True)
        return 1
    actual = srv.server_address[1]
    url = f"http://127.0.0.1:{actual}/"
    print(f"ffind Web 界面: {url}   (Ctrl-C 退出)\n索引文件: {idx.path}\n搜索根:   {', '.join(rs)}", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nffind: 已退出。")
    finally:
        srv.server_close()
    return 0
