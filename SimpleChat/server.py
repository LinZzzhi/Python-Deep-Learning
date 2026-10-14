#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SimpleChat —— 极简在线聊天室（Python 后端）

只用标准库实现（http.server），**没有任何第三方依赖**：
    - 单端口 HTTP 服务，监听环境变量 $PORT，绑定 0.0.0.0
    - 同时提供静态前端（static/）和 JSON API
    - 房间 / 成员 / 消息全部放在内存里，进程重启即清空
    - 管理员账号会尽力持久化到 admin.json（目录不可写时退化成内存态）

为什么前端用短轮询而不是 WebSocket：
    线上是「反向代理 + 单端口」的部署形态。WebSocket 需要 Upgrade 握手、
    SSE 需要长连接不缓冲，这两点都取决于代理配置，这里无法预先验证；
    而普通 HTTP 请求（GET/POST）在任何代理下都一定能通。所以选短轮询，
    代价是最多 1 秒的延迟 —— 对一个几十人的简易聊天室完全够用。

关于「在线」的判定：不做主动下线通知，只靠轮询心跳（ONLINE_TIMEOUT）。
好处是刷新 / 切网络 / 关页面再打开都能自然恢复会话；代价是关掉页面后
最长 15 秒才会从在线列表里消失。

运行：
    python server.py            # 默认 8000 端口
    PORT=3000 python server.py
"""

import hmac
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlparse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
ADMIN_FILE = os.path.join(BASE_DIR, "admin.json")

# ---------------------------------------------------------------- 参数

MAX_MEMBERS = 10            # 每个聊天室的人数上限
MAX_ROOM_NAME = 20          # 聊天室名称长度上限
MAX_USER_NAME = 16          # 昵称长度上限
MAX_TEXT = 500              # 单条消息长度上限
ONLINE_TIMEOUT = 15.0       # 超过这么久没轮询就算离线（前端 1s 轮询一次）
ROOM_EMPTY_TTL = 300.0      # 空房间保留多久后回收（回收时写入历史）
MAX_MESSAGES = 200          # 每个房间保留最近多少条消息
MAX_PARTICIPANTS = 200      # 每个房间最多记录多少位「来过的人」
PASSWORD_LEN = 6            # 聊天室密码位数

ADMIN_USER = "admin"                                # 管理员用户名
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "12345678")
ADMIN_TTL = 2 * 3600        # 管理员登录凭证有效期
HISTORY_MAX = 100           # 历史房间最多保留多少条

_lock = threading.RLock()
_rooms = {}                 # room_id -> Room（当前还活着的房间）
_history = []               # 已解散 / 已回收房间的快照（含成员与消息）

_admin = {"user": ADMIN_USER, "password": ADMIN_PASSWORD}
_admin_tokens = {}          # token -> 过期时间戳


# ---------------------------------------------------------------- 数据模型

class Member(object):
    __slots__ = ("token", "mid", "name", "seen", "joined", "ip", "ghost", "admin")

    def __init__(self, token, name, ip="", ghost=False, admin=False):
        self.token = token          # 会话凭证，只有本人知道
        self.mid = _rand_hex(8)     # 对外公开的成员号（用于移出成员，不能拿来冒充）
        self.name = name
        self.seen = time.time()
        self.joined = time.time()
        self.ip = ip                # 客户端 IP（只在管理员视角下发）
        self.ghost = ghost          # 无痕成员：不计人数、不发加入/离开提示
        self.admin = admin          # 管理员：在房间内拥有房主权限


class Room(object):
    __slots__ = ("id", "name", "password", "owner_token", "owner_name", "created",
                 "members", "messages", "participants", "next_msg_id",
                 "empty_since", "ended")

    def __init__(self, room_id, name, password, owner_token, owner_name):
        self.id = room_id
        self.name = name
        self.password = password        # 6 位数字字符串；None 表示无密码
        self.owner_token = owner_token  # 创建者（拥有全部权限）
        self.owner_name = owner_name    # 创建者昵称（用于历史记录）
        self.created = time.time()
        self.members = {}               # token -> Member
        self.messages = []
        self.participants = []          # 来过的人（含已离开的），历史记录用
        self.next_msg_id = 1
        self.empty_since = None
        self.ended = None


# ---------------------------------------------------------------- 工具函数

def _rand_hex(n):
    return "".join(random.choice("0123456789abcdef") for _ in range(n))


def _new_room_id():
    while True:
        rid = _rand_hex(8)
        if rid not in _rooms:
            return rid


def _new_password():
    """生成一个跟现有聊天室都不重复的 6 位数字密码。"""
    used = set(r.password for r in _rooms.values() if r.password)
    if len(used) >= 10 ** PASSWORD_LEN:
        return None
    while True:
        code = "%0*d" % (PASSWORD_LEN, random.randint(0, 10 ** PASSWORD_LEN - 1))
        if code not in used:
            return code


def _online(room, include_ghosts=False):
    """在线成员。无痕成员默认不算进去（所以人数与成员列表都不受其影响）。"""
    now = time.time()
    return [m for m in room.members.values()
            if now - m.seen <= ONLINE_TIMEOUT and (include_ghosts or not m.ghost)]


def _can_manage(room, member):
    """房主，或者无痕管理员，都能管理房间。"""
    return member.token == room.owner_token or member.admin


def _room_public(room):
    online = _online(room)
    return {
        "id": room.id,
        "name": room.name,
        "count": len(online),
        "max": MAX_MEMBERS,
        "has_password": bool(room.password),
    }


def _member_public(room, member, admin_view=False):
    item = {
        "mid": member.mid,
        "name": member.name,
        "is_owner": _can_manage(room, member),
        "is_you": False,
        "ghost": bool(member.ghost),
    }
    if admin_view:
        item["ip"] = member.ip
        item["joined"] = member.joined
    return item


def _add_participant(room, member):
    room.participants.append({
        "token": member.token,
        "name": member.name,
        "ip": member.ip,
        "joined": member.joined,
        "left": None,
        "ghost": bool(member.ghost),
    })
    if len(room.participants) > MAX_PARTICIPANTS:
        del room.participants[:-MAX_PARTICIPANTS]


def _mark_left(room, token):
    now = time.time()
    for item in room.participants:
        if item["token"] == token and item["left"] is None:
            item["left"] = now


def _sys_msg(room, text):
    """往房间里插一条系统消息（加入 / 离开 / 被移出）。"""
    room.messages.append({
        "id": room.next_msg_id,
        "mid": None,
        "user": None,
        "text": text,
        "ts": time.time(),
        "system": True,
        "ip": None,
    })
    room.next_msg_id += 1
    _trim(room)


def _trim(room):
    if len(room.messages) > MAX_MESSAGES:
        del room.messages[:-MAX_MESSAGES]


def _room_snapshot(room, ended=False):
    """给管理员看的房间全量快照（当前房间 + 历史房间通用）。"""
    return {
        "id": room.id,
        "name": room.name,
        "password": room.password,
        "has_password": bool(room.password),
        "owner": room.owner_name,
        "created": room.created,
        "ended": room.ended if ended else None,
        "active": not ended,
        "online": len(_online(room)),
        "max": MAX_MEMBERS,
        "participants": [
            {"name": p["name"], "ip": p["ip"], "joined": p["joined"],
             "left": p["left"], "ghost": p["ghost"]}
            for p in room.participants
        ],
        "messages": [dict(m) for m in room.messages],
    }


def _archive_room(room):
    """房间结束（解散 / 回收）时，把快照写进历史。"""
    room.ended = time.time()
    _history.append(_room_snapshot(room, ended=True))
    if len(_history) > HISTORY_MAX:
        del _history[:-HISTORY_MAX]


def _purge_locked():
    """清理掉线成员，回收长时间没人的空房间（回收前写入历史）。"""
    now = time.time()
    for rid in list(_rooms):
        room = _rooms[rid]
        for token in [t for t, m in room.members.items()
                      if now - m.seen > ONLINE_TIMEOUT]:
            _mark_left(room, token)
            del room.members[token]
        if room.members:
            room.empty_since = None
        elif room.empty_since is None:
            room.empty_since = now
        elif now - room.empty_since > ROOM_EMPTY_TTL:
            _archive_room(room)
            del _rooms[rid]


def _clean(text, limit):
    """去掉首尾空白和不可见控制字符，并截断长度。"""
    text = "".join("" if ord(ch) < 32 or ord(ch) == 127 else ch
                   for ch in str(text or ""))
    return text.strip()[:limit]


# ---------------------------------------------------------------- 管理员账号

def _load_admin():
    try:
        with open(ADMIN_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        user = _clean(data.get("user"), MAX_USER_NAME)
        password = _clean(data.get("password"), 64)
        if user and password:
            _admin["user"] = user
            _admin["password"] = password
    except Exception:
        pass                        # 首次运行没有文件，或目录不可写


def _save_admin():
    """尽力持久化；沙箱目录只读时就只在内存里生效。"""
    try:
        tmp = ADMIN_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(_admin, fh, ensure_ascii=False)
        os.replace(tmp, ADMIN_FILE)
    except Exception:
        pass


def _admin_ok(token):
    token = token or ""
    expire = _admin_tokens.get(token)
    if not expire or expire < time.time():
        _admin_tokens.pop(token, None)
        return False
    _admin_tokens[token] = time.time() + ADMIN_TTL
    return True


def _admin_drop_tokens():
    now = time.time()
    for token in [t for t, exp in _admin_tokens.items() if exp < now]:
        del _admin_tokens[token]


# ---------------------------------------------------------------- HTTP 服务

class Handler(BaseHTTPRequestHandler):
    server_version = "SimpleChat/1.1"
    protocol_version = "HTTP/1.1"       # 开启 keep-alive，1 秒一次轮询更省

    # ---- 基础输出

    def log_message(self, fmt, *args):
        pass                            # 轮询太频繁，不打日志

    def _send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _fail(self, message, **extra):
        payload = {"ok": False, "error": message}
        payload.update(extra)
        self._send_json(payload)

    def _client_ip(self):
        """线上是反向代理，优先取转发头；本地直连时退回 socket 地址。"""
        for header in ("X-Forwarded-For", "X-Real-IP", "CF-Connecting-IP"):
            value = self.headers.get(header)
            if value:
                return _clean(value.split(",")[0], 45)
        return _clean(self.client_address[0] if self.client_address else "", 45)

    def _send_static(self, rel):
        path = os.path.normpath(os.path.join(STATIC_DIR, rel))
        if not (path == STATIC_DIR or path.startswith(STATIC_DIR + os.sep)) \
                or not os.path.isfile(path):
            self._send_json({"ok": False, "error": "not found"}, 404)
            return
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "application/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".svg": "image/svg+xml",
            ".ico": "image/x-icon",
        }.get(os.path.splitext(path)[1].lower(), "application/octet-stream")
        with open(path, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self):
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return {}
        if length <= 0 or length > 1000000:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            return {}

    # ---- 路由

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path.startswith("/api/"):
            self._api_get(path, parse_qs(parsed.query))
            return
        rel = "index.html" if path in ("/", "") else unquote(path.lstrip("/"))
        self._send_static(rel)

    def do_POST(self):
        self._api_post(urlparse(self.path).path, self._read_json())

    # ---- 取房间和成员（做权限判断用）

    def _lookup(self, room_id, token):
        room = _rooms.get(room_id or "")
        if room is None:
            return None, None
        return room, room.members.get(token or "")

    # ---- GET

    def _api_get(self, path, query):
        if path == "/api/rooms":
            keyword = _clean(query.get("q", [""])[0], 40).lower()
            with _lock:
                _purge_locked()
                rooms = [_room_public(r) for r in _rooms.values() if _online(r)]
            rooms = [r for r in rooms if r["count"] > 0]
            if keyword:
                rooms = [r for r in rooms if keyword in r["name"].lower()]
            rooms.sort(key=lambda r: (-r["count"], r["name"]))
            self._send_json({"ok": True, "rooms": rooms, "max": MAX_MEMBERS})
            return

        if path == "/api/rooms/state":
            self._room_state(query)
            return

        if path == "/api/admin/data":
            self._admin_data(query.get("token", [""])[0])
            return

        self._send_json({"ok": False, "error": "not found"}, 404)

    def _room_state(self, query):
        room_id = query.get("room_id", [""])[0]
        token = query.get("token", [""])[0]
        try:
            since = int(query.get("since", ["0"])[0])
        except ValueError:
            since = 0
        with _lock:
            _purge_locked()
            room, member = self._lookup(room_id, token)
            if room is None:
                self._send_json({"ok": False, "gone": True,
                                 "error": "聊天室已解散"})
                return
            if member is None:
                self._send_json({"ok": False, "removed": True,
                                 "error": "你已不在该聊天室内"})
                return
            member.seen = time.time()
            admin_view = bool(member.admin)
            messages = [dict(m) for m in room.messages if m["id"] > since]
            if not admin_view:
                # 普通用户看不到 IP
                for m in messages:
                    m.pop("ip", None)
            members = []
            me = None
            for m in _online(room, include_ghosts=admin_view):
                item = _member_public(room, m, admin_view)
                if m.token == member.token:
                    item["is_you"] = True
                    me = item
                members.append(item)
            if me is None:                      # 无痕成员不在在线列表里也正常
                me = _member_public(room, member, admin_view)
                me["is_you"] = True
            payload = {
                "ok": True,
                "room": _room_public(room),
                "you": me,
                "members": members,
                "messages": messages,
                "admin_view": admin_view,
                "password": room.password if _can_manage(room, member) else None,
            }
        self._send_json(payload)

    # ---- POST

    def _api_post(self, path, data):
        if path == "/api/rooms":
            self._create_room(data)
        elif path == "/api/rooms/join":
            self._join_room(data)
        elif path == "/api/rooms/send":
            self._send_message(data)
        elif path == "/api/rooms/leave":
            self._leave_room(data)
        elif path == "/api/rooms/kick":
            self._kick_member(data)
        elif path == "/api/rooms/dissolve":
            self._dissolve_room(data)
        elif path == "/api/admin/login":
            self._admin_login(data)
        elif path == "/api/admin/update":
            self._admin_update(data)
        elif path == "/api/admin/ghost":
            self._admin_ghost(data)
        else:
            self._send_json({"ok": False, "error": "not found"}, 404)

    # ---- 房间

    def _create_room(self, data):
        name = _clean(data.get("name"), MAX_ROOM_NAME)
        user = _clean(data.get("user"), MAX_USER_NAME)
        use_password = bool(data.get("password", True))
        if not name:
            self._fail("请填写聊天室名称")
            return
        if not user:
            self._fail("请填写你的昵称")
            return
        with _lock:
            _purge_locked()
            room_id = _new_room_id()
            token = _rand_hex(16)
            room = Room(room_id, name, _new_password() if use_password else None,
                        token, user)
            member = Member(token, user, self._client_ip())
            room.members[token] = member
            _add_participant(room, member)
            _rooms[room_id] = room
            _sys_msg(room, "%s 创建了聊天室" % user)
            self._send_json({
                "ok": True,
                "room_id": room_id,
                "token": token,
                "is_owner": True,
                "password": room.password,
                "room": _room_public(room),
            })

    def _join_room(self, data):
        user = _clean(data.get("user"), MAX_USER_NAME)
        room_id = _clean(data.get("room_id"), 32)
        password = _clean(data.get("password"), 12)
        if not user:
            self._fail("请填写你的昵称")
            return
        with _lock:
            _purge_locked()
            if room_id:
                room = _rooms.get(room_id)
                if room is None:
                    self._fail("聊天室不存在或已解散")
                    return
                if room.password and password != room.password:
                    self._fail("密码不正确")
                    return
            elif password:
                room = None
                for candidate in _rooms.values():
                    if candidate.password == password and _online(candidate):
                        room = candidate
                        break
                if room is None:
                    self._fail("没有找到使用该密码的聊天室")
                    return
            else:
                self._fail("请选择聊天室或输入密码")
                return

            if len(_online(room)) >= MAX_MEMBERS:
                self._fail("聊天室人数已满（最多 %d 人）" % MAX_MEMBERS)
                return

            token = _rand_hex(16)
            member = Member(token, user, self._client_ip())
            room.members[token] = member
            _add_participant(room, member)
            room.empty_since = None
            # 创建者关掉页面再回来时，前端会带上当初的会话凭证，权限跟着人走
            resume = _clean(data.get("resume_token"), 64)
            if resume and resume == room.owner_token:
                room.owner_token = token
            _sys_msg(room, "%s 加入了聊天室" % user)
            self._send_json({
                "ok": True,
                "room_id": room.id,
                "token": token,
                "is_owner": token == room.owner_token,
                "password": room.password if token == room.owner_token else None,
                "room": _room_public(room),
            })

    def _send_message(self, data):
        text = _clean(data.get("text"), MAX_TEXT)
        if not text:
            self._fail("消息不能为空")
            return
        with _lock:
            room, member = self._lookup(data.get("room_id"), data.get("token"))
            if room is None:
                self._fail("聊天室已解散", gone=True)
                return
            if member is None:
                self._fail("你已不在该聊天室内", removed=True)
                return
            member.seen = time.time()
            msg = {"id": room.next_msg_id, "mid": member.mid, "user": member.name,
                   "text": text, "ts": time.time(), "system": False,
                   "ip": member.ip}
            room.next_msg_id += 1
            room.messages.append(msg)
            _trim(room)
            self._send_json({"ok": True, "id": msg["id"]})

    def _leave_room(self, data):
        with _lock:
            room, member = self._lookup(data.get("room_id"), data.get("token"))
            if room is None or member is None:
                self._send_json({"ok": True})
                return
            name = member.name
            ghost = member.ghost
            _mark_left(room, member.token)
            del room.members[member.token]
            if not ghost:               # 无痕进出的痕迹一并抹掉
                _sys_msg(room, "%s 离开了聊天室" % name)
            if not room.members:
                room.empty_since = time.time()
            self._send_json({"ok": True})

    def _kick_member(self, data):
        target_mid = _clean(data.get("target"), 16)
        with _lock:
            room, member = self._lookup(data.get("room_id"), data.get("token"))
            if room is None or member is None:
                self._fail("你已不在该聊天室内")
                return
            if not _can_manage(room, member):
                self._fail("只有聊天室创建者才能移出成员")
                return
            victim = None
            for m in room.members.values():
                if m.mid == target_mid:
                    victim = m
                    break
            if victim is None:
                self._fail("该成员已不在房间内")
                return
            if victim.token == member.token:
                self._fail("不能移出自己")
                return
            _mark_left(room, victim.token)
            del room.members[victim.token]
            if not victim.ghost:
                _sys_msg(room, "%s 被移出聊天室" % victim.name)
            self._send_json({"ok": True})

    def _dissolve_room(self, data):
        with _lock:
            room, member = self._lookup(data.get("room_id"), data.get("token"))
            if room is None or member is None:
                self._fail("你已不在该聊天室内")
                return
            if not _can_manage(room, member):
                self._fail("只有聊天室创建者才能解散聊天室")
                return
            _archive_room(room)         # 解散也进历史，消息内容不丢
            del _rooms[room.id]
            self._send_json({"ok": True})

    # ---- 管理员

    def _need_admin(self, token):
        if _admin_ok(token):
            return True
        self._fail("管理员登录已过期，请重新登录", admin_expired=True)
        return False

    def _admin_login(self, data):
        user = _clean(data.get("user"), MAX_USER_NAME)
        password = _clean(data.get("password"), 64)
        with _lock:
            _admin_drop_tokens()
            ok_user = hmac.compare_digest(user.encode("utf-8"),
                                          _admin["user"].encode("utf-8"))
            ok_pwd = hmac.compare_digest(password.encode("utf-8"),
                                         _admin["password"].encode("utf-8"))
            if not (ok_user and ok_pwd):
                self._fail("管理员用户名或密码不正确")
                return
            token = _rand_hex(24)
            _admin_tokens[token] = time.time() + ADMIN_TTL
            self._send_json({"ok": True, "token": token, "user": _admin["user"]})

    def _admin_update(self, data):
        with _lock:
            if not self._need_admin(data.get("token")):
                return
            user = _clean(data.get("user"), MAX_USER_NAME)
            password = _clean(data.get("password"), 64)
            if not user:
                self._fail("管理员用户名不能为空")
                return
            if len(password) < 4:
                self._fail("管理员密码至少 4 位")
                return
            _admin["user"] = user
            _admin["password"] = password
            _save_admin()
            self._send_json({"ok": True, "user": user, "password": password})

    def _admin_data(self, token):
        with _lock:
            _purge_locked()
            if not self._need_admin(token):
                return
            rooms = [_room_snapshot(r) for r in _rooms.values()]
            rooms.extend(_history)
            rooms.sort(key=lambda r: -r["created"])
            users = {}
            for room in rooms:
                for p in room["participants"]:
                    users.setdefault(p["name"], p["ip"])
            self._send_json({
                "ok": True,
                "user": _admin["user"],
                "password": _admin["password"],
                "rooms": rooms,
                "stats": {
                    "rooms": len(rooms),
                    "active": len(_rooms),
                    "messages": sum(len(r["messages"]) for r in rooms),
                    "users": len(users),
                },
                "users": [{"name": n, "ip": ip} for n, ip in sorted(users.items())],
            })

    def _admin_ghost(self, data):
        """无痕加入：不校验密码、不占名额、不计人数、不发加入提示，但有房主权限。"""
        with _lock:
            _purge_locked()
            if not self._need_admin(data.get("token")):
                return
            room = _rooms.get(_clean(data.get("room_id"), 32))
            if room is None:
                self._fail("聊天室不存在或已解散")
                return
            token = _rand_hex(16)
            member = Member(token, _admin["user"], self._client_ip(),
                            ghost=True, admin=True)
            room.members[token] = member
            _add_participant(room, member)
            room.empty_since = None
            self._send_json({
                "ok": True,
                "room_id": room.id,
                "token": token,
                "ghost": True,
                "is_owner": True,
                "room": _room_public(room),
            })


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    _load_admin()
    port = int(os.environ.get("PORT", "8000"))
    httpd = Server(("0.0.0.0", port), Handler)
    print("SimpleChat 已启动: http://0.0.0.0:%d" % port, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
