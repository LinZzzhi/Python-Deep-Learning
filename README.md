# Python 语言与深度学习 · 课程大作业仓库

本仓库是《Python 语言与深度学习》课程大作业的代码集合，包含两个彼此独立、均可直接运行的
小项目：

| 项目 | 一句话说明 | 技术栈 |
| --- | --- | --- |
| [**SimpleChat**](#项目一simplechat--极简在线聊天室) | 极简在线聊天室，支持创建 / 搜索 / 凭密码加入房间与管理员控制台 | Python 标准库（`http.server`）+ 原生 JavaScript |
| [**SimplePlayer**](#项目二simpleplayer--简易音视频播放器) | 深色扁平风格的本地音视频播放器，支持拖放打开 | Tkinter + ffpyplayer（静态内嵌 FFmpeg） |

两个项目都尽量做到**零/极少第三方依赖**，并且不依赖系统额外安装运行时组件。

## 目录结构

```
Python深度学习/
├── README.md
├── SimpleChat/                 # 项目一：在线聊天室
│   ├── server.py               # 后端（纯标准库，单端口 HTTP 服务 + JSON API）
│   ├── requirements.txt        # 无第三方依赖（仅作说明）
│   └── static/                 # 前端静态资源，由 server.py 一并托管
│       ├── index.html
│       ├── app.js
│       └── style.css
└── SimplePlayer/               # 项目二：音视频播放器
    ├── player.py               # 主程序（GUI + 播放逻辑）
    ├── requirements.txt        # ffpyplayer、tkinterdnd2
    ├── run.bat                 # 双击启动器
    └── .venv/                  # 本地虚拟环境（不纳入版本管理）
```

---

## 项目一：SimpleChat —— 极简在线聊天室

### 功能特性

- **大厅**：列出当前有人的聊天室（名称、在线人数、是否设密码），支持关键字实时搜索；
- **创建房间**：可自动生成**全局不重复的 6 位数字进房密码**，也可选择不设密码；每个房间最多 10 人；
- **加入房间**：列表点「加入」，或直接输入 6 位密码加入；
- **房主权限**：移出成员、解散聊天室；关掉浏览器再回来权限不丢（凭证存在本地，重新进入时自动恢复）；
- **消息**：房间内公共聊天，系统消息（创建 / 加入 / 离开 / 被移出）与用户消息同一消息流；
- **管理员控制台**：昵称填 `admin` 后输入管理员密码进入，可查看所有房间（含已结束房间）的
  参与者用户名 / IP / 进出时间与全部消息，可修改管理员账号，并可**无痕加入**任意房间。

### 技术选型说明

- **后端只用标准库**（`http.server.ThreadingHTTPServer`），没有第三方依赖，避免部署时安装失败；
- **前端用短轮询（1 秒）而非 WebSocket / SSE**：线上是「反向代理 + 单端口」形态，WebSocket 需要
  Upgrade 握手、SSE 需要长连接不缓冲，都依赖代理配置；普通 GET/POST 在任何代理下都能通。
  代价是最多 1 秒延迟，对几十人的简易聊天室完全够用；
- **房间 / 成员 / 消息全部放在内存里**，进程重启即清空（符合「临时聊天室」的预期）。

### 快速开始

```bash
cd SimpleChat

# 默认 8000 端口，绑定 0.0.0.0
python server.py

# 或指定端口
PORT=3000 python server.py
```

启动后浏览器打开 `http://127.0.0.1:8000` 即可。同一局域网内其他设备访问
`http://<本机 IP>:8000` 也能使用。

同时项目已部署到公网，可直接使用浏览器访问。

### 配置项

环境变量：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PORT` | `8000` | HTTP 服务监听端口 |
| `ADMIN_PASSWORD` | `12345678` | 管理员初始密码（登录后可在控制台内修改） |

主要业务参数（集中在 `server.py` 顶部，按需修改）：

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `MAX_MEMBERS` | 10 | 每个房间在线人数上限 |
| `MAX_ROOM_NAME` / `MAX_USER_NAME` | 20 / 16 | 房间名 / 昵称长度上限 |
| `MAX_TEXT` | 500 | 单条消息长度上限 |
| `ONLINE_TIMEOUT` | 15.0 s | 超过这么久没有轮询即视为离线 |
| `ROOM_EMPTY_TTL` | 300.0 s | 空房间保留时长，超时回收并写入历史 |
| `MAX_MESSAGES` / `MAX_PARTICIPANTS` | 200 / 200 | 每个房间保留的消息数 / 访客记录数 |
| `PASSWORD_LEN` | 6 | 房间密码位数 |
| `ADMIN_TTL` / `HISTORY_MAX` | 2 h / 100 | 管理员登录有效期 / 历史房间保留条数 |

### 后端 API

所有接口都以 JSON 收发，成功返回 `{"ok": true, ...}`，失败返回 `{"ok": false, "error": "..."}`。

| 方法 | 路径 | 主要参数 | 说明 |
| --- | --- | --- | --- |
| GET | `/api/rooms` | `q`（可选搜索词） | 房间列表（仅含有在线成员的房间） |
| GET | `/api/rooms/state` | `room_id`、`token`、`since` | 房间状态与增量消息（`since` 为已收到的最大消息 id） |
| POST | `/api/rooms` | `name`、`user`、`password`（是否启用密码） | 创建房间，返回 `room_id` 与 `token` |
| POST | `/api/rooms/join` | `user`、`room_id` 或 `password`、`resume_token`（可选） | 加入房间 |
| POST | `/api/rooms/send` | `room_id`、`token`、`text` | 发送消息 |
| POST | `/api/rooms/leave` | `room_id`、`token` | 主动离开（前端用 `sendBeacon` 调用） |
| POST | `/api/rooms/kick` | `room_id`、`token`、`target`（成员 `mid`） | 移出成员（房主 / 管理员） |
| POST | `/api/rooms/dissolve` | `room_id`、`token` | 解散房间（房主 / 管理员） |
| POST | `/api/admin/login` | `user`、`password` | 管理员登录，返回管理员 `token` |
| POST | `/api/admin/update` | `token`、`user`、`password` | 修改管理员账号 |
| POST | `/api/admin/ghost` | `token`、`room_id` | 以无痕身份加入房间 |
| GET | `/api/admin/data` | `token` | 管理员数据总览（房间、访客、消息、统计） |

### 设计与实现要点

- **成员有两套标识**：`token`（会话凭证，只有本人知道，用于鉴权）与 `mid`（对外公开，
  仅用于「移出成员」），避免把凭证泄露给同房间的人。判断「这条消息是不是我发的」用 `mid`
  而不是昵称（昵称可重复）。
- **在线判定**：15 秒内有轮询即在线；点「离开」时用 `navigator.sendBeacon` 立即通知服务端，
  不用等超时。**不在 `beforeunload` 里主动下线**，否则刷新页面会先把自己删掉、重新加载后被
  踢回大厅；在线状态交给心跳自然收敛。
- **创建者权限跟随会话**：前端把创建凭证写入 `localStorage`，再次加入时作为 `resume_token`
  提交，服务端校验后转移所有权，因此关掉浏览器再回来权限不丢。
- **前端轮询串行化**：任何时刻只允许一个 `/api/rooms/state` 请求在飞（`pollBusy` 锁 +
  `pollAgain` 补拉），并以已渲染消息 id 集合做幂等去重，避免同一批消息被画两遍；
  下一次轮询的时间按「上一轮开始时刻 + 间隔」计算，抵消请求耗时造成的漂移。
- **安全**：消息渲染全部使用 `textContent`（防 XSS）；静态文件路径做规范化校验（防路径穿越）；
  管理员密码比较使用 `hmac.compare_digest`。
- **账号持久化**：管理员账号会尽力写入 `SimpleChat/admin.json`；目录不可写时自动退化为内存态。
- **IP 采集**：优先取 `X-Forwarded-For` → `X-Real-IP` → socket 地址，**只在管理员视角下发**，
  普通用户拿不到任何 IP 字段。

> 说明：`index.html` 中的提示文案写的是「默认密码 12345678」，而 `server.py` 的默认值是
> `ADMIN_PASSWORD` 环境变量未设置时的 `12345678`。两者不一致，请以 `server.py` 为准，或自行统一。

---

## 项目二：SimplePlayer —— 简易音视频播放器

### 功能特性

- 本地音视频播放：打开文件、播放 / 暂停 / 停止、进度条拖动定位、音量条调节；
- **拖放打开**：把音视频文件拖到窗口任意位置即可播放（依赖 `tkinterdnd2`，装不上会自动降级为
  只能用「打开文件」按钮）；
- **深色扁平界面**：按钮、进度条、音量条全部用 Canvas 自绘，图标为矢量绘制，
  不依赖任何第三方 UI 库或图标字体；
- 窗口缩放时视频画面等比自适应（缩放交由 FFmpeg 的 `swscale` 完成，效率高），
  纯音频文件显示播放提示而非黑屏。

### 技术栈

- **Tkinter**：图形界面，Python 自带；
- **ffpyplayer**：解码与播放，该库**静态内嵌了 FFmpeg**（libav\*/swscale/swresample 全部编译进
  扩展模块），因此本机**不需要另外安装 ffmpeg**；
- 无需 Pillow：Tk 8.6 原生支持 PPM(P6)，可直接把原始 RGB 数据显示为 `PhotoImage`。

### 环境准备

> 前提：所使用的 Python 必须带 `tkinter`。部分精简版 Python 不带（构建 GUI 会直接失败）。

```bash
cd SimplePlayer

python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

`requirements.txt`：

```
ffpyplayer==4.5.3
tkinterdnd2==0.6.3
```

### 运行

- **双击 `run.bat`**（会用本目录的 `.venv` 启动，也可把媒体文件直接拖到 `run.bat` 上打开）；
- 或命令行：

```bash
.venv\Scripts\python.exe player.py [媒体文件]
```

### 操作说明

| 操作 | 方式 |
| --- | --- |
| 打开文件 | 点击左下角「打开文件」，或把文件拖入窗口 |
| 播放 / 暂停 | 中间圆形主按钮（图标随状态切换） |
| 停止 | 主按钮右侧的停止按钮 |
| 定位 | 拖动或点击上方进度条，松手后才真正跳转 |
| 调节音量 | 右下角音量条 |

### 支持的格式

由内嵌的 FFmpeg 决定，常见格式均可播放：

- **视频**：mp4 / mkv / avi / mov / flv / webm / wmv / ts / m4v / mpg / mpeg / 3gp / rmvb / vob / ogv / m2ts / asf
- **音频**：mp3 / flac / wav / aac / m4a / ogg / oga / opus / wma / ape / alac / amr / mp2 / ac3 / dts / mid / aiff

### 设计与实现要点

- **元数据需要反复同步**：`duration` 由读线程在打开阶段就填好，而视频源尺寸 `src_vid_size`
  要等视频线程解出第一帧才填（实测相差 100 ms 以上）。因此不能一拿到 `duration` 就断定
  「没有视频流 = 纯音频」，否则大文件会被永久误判成音频。程序改为只要还没确认存在视频流就持续
  复查，「纯音频」结论要等 1 秒观察期过后才下。
- **不能被调用的接口**：文件真正打开之前调用 `get_pts()` 会导致进程崩溃（FFmpeg 内部空指针），
  所以用 `_ready` 门控，只有在 `duration` 被填充后才允许调用。
- **暂停状态下的定位**：`get_frame()` 在暂停时恒返回 `(None, 'paused')`，定位不会刷新画面。
  做法是临时解除暂停取到目标帧后再暂停回去（`_scrubbing` 标记）。
- **定位后的旧帧**：解码器里可能残留定位前的帧，需要按目标时间戳容差丢弃；丢弃必须
  **批量进行**（`while` 循环，上限 500 帧），因为 Tk 定时器精度约 15 ms，一帧挂一个回调会
  因积压而超时、停在错误的帧上。
- **进度显示**：有视频时以「屏幕上已显示那一帧的时间戳」为准，保证进度条与画面一致；
  定位后 FFmpeg 需要几秒把画面拉回音频主时钟，这期间直接使用音频时钟会与画面明显对不上。

---

## 环境与依赖汇总

| 项目 | Python | 第三方依赖 | 其他要求 |
| --- | --- | --- | --- |
| SimpleChat | 3.x（标准库即可） | **无** | 无需前端构建，静态文件由后端一并托管 |
| SimplePlayer | 需带 `tkinter` | `ffpyplayer==4.5.3`、`tkinterdnd2==0.6.3` | 已内嵌 FFmpeg，无需单独安装 |

## 备注

- `SimplePlayer/.venv/` 是本地虚拟环境，**不纳入版本管理**，克隆后请按上文重新创建。
- SimpleChat 已按「单端口 + 反向代理」的形态设计，可直接部署到支持 `$PORT` 环境变量的托管环境
  （启动命令 `python server.py`）。
