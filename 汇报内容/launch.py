"""Local presentation launcher. Uses Python standard library only."""
from __future__ import annotations
import sys
sys.dont_write_bytecode = True
import argparse
import functools
import json
import os
from pathlib import Path
import secrets
import subprocess
import threading
import time
import urllib.request
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNTIME = HERE / 'runtime'
STATE = RUNTIME / 'session.json'

def request(url, data=None, token=''):
    req = urllib.request.Request(url, data=data, headers={'X-Presentation-Token': token})
    with urllib.request.urlopen(req, timeout=2) as response:
        return json.load(response)

def existing():
    try:
        state = json.loads(STATE.read_text('utf-8'))
        health = request(state['url'] + 'api/health')
        if health.get('instance') == state['token']:
            return state
    except (OSError, ValueError, KeyError):
        pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stop', action='store_true')
    parser.add_argument('--no-player', action='store_true')
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    state = existing()
    if args.stop:
        if state:
            request(state['url'] + 'api/stop', b'{}', state['token'])
            print('Presentation stopped. The player can be closed separately.')
        else:
            print('Presentation is not running.')
        return
    if state:
        if not args.no_browser:
            webbrowser.open(state['url'])
        print('Already running: ' + state['url'])
        return
    RUNTIME.mkdir(exist_ok=True)
    if args.background:
        with (RUNTIME / 'launcher.log').open('ab') as log:
            child = subprocess.Popen([sys.executable, str(HERE / 'launch.py')], cwd=HERE, stdout=log, stderr=log, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        for _ in range(80):
            if existing():
                print('Presentation is ready.')
                return
            if child.poll() is not None:
                break
            time.sleep(.25)
        raise RuntimeError('Startup failed. See runtime/launcher.log.')
    sys.path.insert(0, str(ROOT / 'file-search'))
    from ffinder.index import Index
    from ffinder.scan import DEFAULT_EXCLUDES
    from ffinder.server import make_server
    idx = Index(str(RUNTIME / 'finder.sqlite'))
    finder = make_server(idx, [str(ROOT)], port=0)
    threading.Thread(target=finder.serve_forever, daemon=True).start()
    token = secrets.token_hex(24)
    player_status = {'ok': False, 'message': '播放器尚未启动'}
    player_process = None
    def start_player():
        nonlocal player_process
        if player_process is not None and player_process.poll() is None:
            return {'ok': True, 'message': '播放器已在运行，请切换到播放器窗口'}
        candidates = [ROOT / 'media-player/.venv/Scripts/pythonw.exe', Path(sys.executable)]
        probe = ROOT / 'media-player/.venv/Scripts/python.exe'
        interpreter = candidates[0] if candidates[0].exists() else candidates[1]
        check = probe if probe.exists() else interpreter
        test = subprocess.run([str(check), '-c', 'import tkinter; from ffpyplayer.player import MediaPlayer'], capture_output=True, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if test.returncode:
            return {'ok': False, 'message': '播放器缺少依赖，详见 runtime/player.log 和使用说明。'}
        with (RUNTIME / 'player.log').open('ab') as log:
            player_process = subprocess.Popen([str(interpreter), str(ROOT / 'media-player/player.py')], cwd=ROOT / 'media-player', stdout=log, stderr=log)
        return {'ok': True, 'message': '播放器已启动，请切换到 SimplePlayer 窗口'}

    class Handler(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass
        def do_GET(self):
            if self.path == '/api/health':
                return self.send_json({'instance': token})
            if self.path == '/runtime.json':
                return self.send_json({'finder': f'http://localhost:{finder.server_port}/', 'player': player_status})
            # Serve presentation assets only; do not expose runtime logs or session token.
            route = self.path.split('?')[0]
            asset = (HERE / route.lstrip('/')).resolve()
            allowed_asset = asset.is_relative_to((HERE / 'assets').resolve()) and asset.is_file()
            if route not in ('/', '/index.html', '/style.css', '/app.js', '/export.html', '/课程设计汇报_长页.pdf') and not allowed_asset:
                return self.send_error(404)
            super().do_GET()
        def send_json(self, value):
            body = json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def do_POST(self):
            if self.path == '/api/stop' and secrets.compare_digest(self.headers.get('X-Presentation-Token', ''), token):
                self.send_json({'ok': True})
                threading.Thread(target=server.shutdown, daemon=True).start()
                return
            origin = self.headers.get('Origin', '')
            if origin not in (f'http://127.0.0.1:{server.server_port}', f'http://localhost:{server.server_port}'):
                return self.send_error(403)
            if self.path == '/api/player':
                player_status.update(start_player())
                return self.send_json(player_status)
            self.send_error(404)

    server = None
    for port in range(8877, 8898):
        try:
            server = ThreadingHTTPServer(('127.0.0.1', port), functools.partial(Handler, directory=str(HERE)))
            break
        except OSError:
            continue
    if server is None:
        finder.shutdown()
        raise RuntimeError('Ports 8877-8897 are occupied.')
    url = f'http://localhost:{server.server_port}/'
    STATE.write_text(json.dumps({'url': url, 'token': token, 'pid': os.getpid()}), encoding='utf-8')
    if not args.no_player:
        player_status.update(start_player())
    # Build the demo index only within the course project; keep all generated data here.
    def build_index():
        if not idx.exists():
            try:
                idx.build([str(ROOT)], workers=2, excludes=DEFAULT_EXCLUDES | {'.venv-cuda', '备份', 'runtime', '.export-work'})
            except Exception as exc:
                (RUNTIME / 'index-error.log').write_text(str(exc), encoding='utf-8')
    threading.Thread(target=build_index, daemon=True).start()
    print('Presentation: ' + url, flush=True)
    print('File finder: ' + f'http://127.0.0.1:{finder.server_port}/', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        finder.shutdown()
        finder.server_close()
        server.server_close()
        STATE.unlink(missing_ok=True)

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()
