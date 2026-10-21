# Python 语言与深度学习 · 课程设计

小组：Coco；汇报人：林子文，许睿泽；组员：万桦平，吴文涛。

## 当前目录与入口

| 目录 | 项目 | 运行入口 |
| --- | --- | --- |
| `snake-ai/` | 强化学习贪吃蛇 | `python webapp.py`（8333）或 `python enhancedSnake.py play` |
| `radar-iq/` | 雷达脉内调制识别 | `python -m aimc.web --port 8080` |
| `web-chat/` | SimpleChat 在线聊天室 | `python server.py`（默认 8000，以控制台输出为准） |
| `file-search/` | SQLite 文件查找器 | `run.bat` 或 `python -m ffinder serve --open`（8765） |
| `media-player/` | SimplePlayer 音视频播放器 | `run.bat` |
| `汇报内容/` | 暗色展示网页、离线导出页与长页 PDF | `一键启动.bat`；`停止展示.bat` |
| `演示素材/` | 用户提供的演示媒体 | 可拖入播放器 |
| `课程设计汇报/` | 既有 PPT 与汇报材料 | 保留原文件 |
| `备份/20261021_整理前/` | 整理前网页、项目源码文档快照与清理记录 | 仅用于恢复 |

五个项目已从 `snake`、`python_homework`、`SimpleChat`、`file-finder`、`SimplePlayer` 改为上表目录。程序导入名保持原样，改动仅涉及目录名称和相应启动路径。

## 展示网页与 PDF

- 双击 `汇报内容/一键启动.bat`，启动本地展示、文件查找器及桌面播放器。网页默认使用 `http://localhost:8877/`，被占用时自动换端口。
- 双击 `汇报内容/停止展示.bat`，停止本组网页与查找服务；播放器窗口可单独关闭。
- 网页中四个 Web 项目保留原版内嵌和全屏演示；播放器区域改为用户提供的截图。桌面播放器仍可由一键启动入口打开。
- `汇报内容/export.html` 是 PDF 对应的离线静态展示页：四个 Web 项目替换为完整展开的截图，避免滚动容器或跨域 iframe 在打印时被裁切。
- `汇报内容/课程设计汇报_长页.pdf` 为单页长 PDF；打开后按宽度适配阅读。静态截图不具备游戏、聊天或识别交互能力。
- 公网演示地址：
  - 贪吃蛇：<https://snake-ai-console.app.workbuddy.host/?sharecode=7EDSNKWLRL>
  - 调制识别：<https://radar-modulation-workbench.app.workbuddy.host/?sharecode=HJIJIXYO66>
  - 聊天室：<https://simplechat-online.app.workbuddy.host/?sharecode=G4RG3DQ4ZB>

## 运行依赖与当前功能

所有命令均在对应项目目录执行。Python 推荐 3.13；项目依赖各自安装，不将本机旧虚拟环境视为可跨平台复制的安装包。

- `snake-ai`：使用 `uv sync` 准备 `pyproject.toml` / `uv.lock` 中的依赖，再用 `uv run python webapp.py`。默认模型推理结合预判规划，不在演示时更新权重。
- `radar-iq`：`python -m pip install -r requirements.txt` 后运行 Web 入口。8 类合成模型输入为 1024 点 / 1 MHz，30 类官方小样本模型为 1000 点 / 50 MHz；现有权重和正式评估图表保留，原始 IQ 数据目录未包含在当前副本中。
- `web-chat`：使用 Python 标准库，数据与房间状态主要保存在内存。
- `file-search`：使用 Python 标准库。单击选中，双击文件夹打开自身，双击文件或按回车只打开所在目录，不执行文件。索引是文件名快照，目录改名后需要重建。
- `media-player`：使用 Tkinter、ffpyplayer、tkinterdnd2。现有 `.venv` 原位保留；新环境用 `python -m pip install -r requirements.txt`。频谱复用 ffpyplayer 安装附带的 FFmpeg，不另加 Python 依赖。支持 WAV、MP3、FLAC、M4A、AAC、OGG、OPUS、WMA 的频谱分析；暂停冻结画面，定位后重算，关闭与切换文件时释放分析进程。

## 整理与恢复

项目级 README 已全部集中到本文档；虚拟环境内第三方包自带的 README、许可证和依赖源码不作清理。正式模型、训练曲线、混淆矩阵及网页使用的时频图保留。测试目录、性能测试脚本、临时音频样例、重复预测结果和缓存已清理；清理记录见备份目录。需要复现下方历史测试命令时，先从 `项目源码与原始文档.zip` 恢复相应测试或示例文件。

以下附录保留原始 README 的技术细节与历史实验记录，目录路径已更新；涉及历史数据下载、旧平台环境或已归档脚本的内容应结合上述当前状态阅读。


<a id="original-overview"></a>

#### 附录 A · 原根目录说明

### Python 语言与深度学习 · 课程大作业仓库

本仓库是《Python 语言与深度学习》课程大作业的代码集合，包含五个彼此独立、均可直接运行的
项目，以及课程设计报告：

| 项目 | 一句话说明 | 技术栈 |
| --- | --- | --- |
| [**SimpleChat**](#项目一simplechat--极简在线聊天室) | 极简在线聊天室，支持创建 / 搜索 / 凭密码加入房间与管理员控制台（已上线） | Python 标准库（`http.server`）+ 原生 JavaScript |
| [**SimplePlayer**](#项目二simpleplayer--简易音视频播放器) | 深色扁平风格的本地音视频播放器，支持拖放打开 | Tkinter + ffpyplayer（静态内嵌 FFmpeg） |
| [**snake**](#项目三snake--基于强化学习的贪吃蛇) | Double DQN 贪吃蛇，含手动游戏、模型演示、哈密顿环必胜策略与 Web 控制台 | PyTorch(CPU) + pygame + FastAPI（uv 管理） |
| [**python_homework**](#项目四python_homework--雷达脉内调制样式识别) | 雷达脉内调制样式识别：IQ → STFT 功率谱 → STFT-CNN → 类别，带中文 Web 工作台 | PyTorch(CPU) + numpy + h5py + Flask |
| [**file-finder**](#项目五file-finder--ffind-文件查找器) | `ffind`：一次性建索引 + SQLite 毫秒级查询的文件名查找器，含 CLI 与本地 Web 界面 | **纯标准库**（Python ≥ 3.10） |

根目录另有《Python编程语言与深度学习课程设计报告_1.docx》，为本次课程设计的总报告。

各项目都尽量做到**零/极少第三方依赖**，并优先使用标准库；每个项目的详细说明见其目录下的
`README.md`，本文档给出总览与关键要点。

#### 目录结构

```
Python深度学习/
├── README.md
├── Python编程语言与深度学习课程设计报告_1.docx   # 课程设计报告
├── .gitignore / .python-version
├── web-chat/                 # 项目一：在线聊天室
│   ├── server.py               # 后端（纯标准库，单端口 HTTP 服务 + JSON API）
│   ├── requirements.txt        # 无第三方依赖（仅作说明）
│   └── static/                 # 前端静态资源，由 server.py 一并托管
│       ├── index.html / app.js / style.css
├── media-player/               # 项目二：音视频播放器
│   ├── player.py               # 主程序（GUI + 播放逻辑）
│   ├── requirements.txt        # ffpyplayer、tkinterdnd2
│   ├── run.bat                 # 双击启动器
│   └── .venv/                  # 本地虚拟环境（不纳入版本管理）
├── snake-ai/                      # 项目三：强化学习贪吃蛇
│   ├── enhancedSnake.py        # 环境 / DQN / 训练 / 推理 / 手动 / 完美策略
│   ├── demo_rl_snake.py        # 播放已有模型的轻量入口
│   ├── webapp.py + web/        # Web 控制台（FastAPI，默认 8333 端口）
│   ├── snake_model.pth         # 已训练权重
│   ├── training_progress.png   # 训练曲线
│   ├── pyproject.toml / uv.lock
│   ├── tests/                  # 环境、模型、训练目标、路径规划、Web 回归测试
│   └── 基于强化学习的贪吃蛇游戏_详细设计文档.pdf / 本科生汇报稿.md
├── radar-iq/            # 项目四：雷达脉内调制样式识别
│   ├── aimc/                   # 信号生成、数据、引擎、上游模型封装、绘图、Web
│   ├── web/                    # 中文 Web 工作台前端（含内联 vendor 的 lucide 图标）
│   ├── third_party/            # 上游 AIMC-image-classification（固定提交，原样保留）
│   ├── tests/                  # 划分、SNR、格式校验、推理一致性、Web
│   ├── requirements.txt / requirements-lock.txt
│   └── data/ 与 outputs/       # 数据与训练产物（不纳入版本管理）
└── file-search/                # 项目五：ffind 文件查找器
    ├── ffinder/                # scan / index / cli / server / web
    ├── tests/                  # 扫描、索引、查询、CLI、Web API
    ├── bench.py                # 对 find 的速度基准
    └── pyproject.toml          # 可选安装，暴露 ffind 命令
```

---

#### 项目一：SimpleChat —— 极简在线聊天室

##### 功能特性

- **大厅**：列出当前有人的聊天室（名称、在线人数、是否设密码），支持关键字实时搜索；
- **创建房间**：可自动生成**全局不重复的 6 位数字进房密码**，也可选择不设密码；每个房间最多 10 人；
- **加入房间**：列表点「加入」，或直接输入 6 位密码加入；
- **房主权限**：移出成员、解散聊天室；关掉浏览器再回来权限不丢（凭证存在本地，重新进入时自动恢复）；
- **消息**：房间内公共聊天，系统消息（创建 / 加入 / 离开 / 被移出）与用户消息同一消息流；
- **管理员控制台**：昵称填 `admin` 后输入管理员密码进入，可查看所有房间（含已结束房间）的
  参与者用户名 / IP / 进出时间与全部消息，可修改管理员账号，并可**无痕加入**任意房间。

##### 技术选型说明

- **后端只用标准库**（`http.server.ThreadingHTTPServer`），没有第三方依赖，避免部署时安装失败；
- **前端用短轮询（1 秒）而非 WebSocket / SSE**：线上是「反向代理 + 单端口」形态，WebSocket 需要
  Upgrade 握手、SSE 需要长连接不缓冲，都依赖代理配置；普通 GET/POST 在任何代理下都能通。
  代价是最多 1 秒延迟，对几十人的简易聊天室完全够用；
- **房间 / 成员 / 消息全部放在内存里**，进程重启即清空（符合「临时聊天室」的预期）。

##### 快速开始

```bash
cd web-chat

# 默认 8000 端口，绑定 0.0.0.0
python server.py

# 或指定端口
PORT=3000 python server.py
```

启动后浏览器打开 `http://127.0.0.1:8000` 即可。同一局域网内其他设备访问
`http://<本机 IP>:8000` 也能使用。

本项目同时已部署到公网，可直接访问：<https://simple-chat-room.app.workbuddy.host/>

##### 配置项

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

##### 后端 API

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

##### 设计与实现要点

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
- **账号持久化**：管理员账号会尽力写入 `web-chat/admin.json`；目录不可写时自动退化为内存态。
- **IP 采集**：优先取 `X-Forwarded-For` → `X-Real-IP` → socket 地址，**只在管理员视角下发**，
  普通用户拿不到任何 IP 字段。

---

#### 项目二：SimplePlayer —— 简易音视频播放器

##### 功能特性

- 本地音视频播放：打开文件、播放 / 暂停 / 停止、进度条拖动定位、音量条调节；
- **拖放打开**：把音视频文件拖到窗口任意位置即可播放（依赖 `tkinterdnd2`，装不上会自动降级为
  只能用「打开文件」按钮）；
- **深色扁平界面**：按钮、进度条、音量条全部用 Canvas 自绘，图标为矢量绘制，
  不依赖任何第三方 UI 库或图标字体；
- 窗口缩放时视频画面等比自适应（缩放交由 FFmpeg 的 `swscale` 完成，效率高），
  纯音频文件显示播放提示而非黑屏。

##### 技术栈

- **Tkinter**：图形界面，Python 自带；
- **ffpyplayer**：解码与播放，该库**静态内嵌了 FFmpeg**（libav\*/swscale/swresample 全部编译进
  扩展模块），因此本机**不需要另外安装 ffmpeg**；
- 无需 Pillow：Tk 8.6 原生支持 PPM(P6)，可直接把原始 RGB 数据显示为 `PhotoImage`。

##### 环境准备

> 前提：所使用的 Python 必须带 `tkinter`。部分精简版 Python 不带（构建 GUI 会直接失败）。

```bash
cd media-player

python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

`requirements.txt`：

```
ffpyplayer==4.5.3
tkinterdnd2==0.6.3
```

##### 运行

- **双击 `run.bat`**（会用本目录的 `.venv` 启动，也可把媒体文件直接拖到 `run.bat` 上打开）；
- 或命令行：

```bash
.venv\Scripts\python.exe player.py [媒体文件]
```

##### 操作说明

| 操作 | 方式 |
| --- | --- |
| 打开文件 | 点击左下角「打开文件」，或把文件拖入窗口 |
| 播放 / 暂停 | 中间圆形主按钮（图标随状态切换） |
| 停止 | 主按钮右侧的停止按钮 |
| 定位 | 拖动或点击上方进度条，松手后才真正跳转 |
| 调节音量 | 右下角音量条 |

##### 支持的格式

由内嵌的 FFmpeg 决定，常见格式均可播放：

- **视频**：mp4 / mkv / avi / mov / flv / webm / wmv / ts / m4v / mpg / mpeg / 3gp / rmvb / vob / ogv / m2ts / asf
- **音频**：mp3 / flac / wav / aac / m4a / ogg / oga / opus / wma / ape / alac / amr / mp2 / ac3 / dts / mid / aiff

##### 设计与实现要点

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

#### 项目三：snake —— 基于强化学习的贪吃蛇

面向教学的单文件 Double DQN 贪吃蛇项目，核心实现集中在 `enhancedSnake.py`，同时提供
手动游戏、已训练模型演示和哈密顿环必胜策略。默认 AI 把模型与安全路径规划结合，
优先走安全的最短食物路线。配套 PDF 描述原始 DQN 设计，**当前算法改进以代码与
`README.md` 为准**。

##### 环境准备

使用 Python 3.13 与 [uv](https://docs.astral.sh/uv/) 管理依赖（CPU 版 PyTorch）：

```bash
cd snake-ai
uv sync
```

依赖：`torch`(CPU) / `numpy` / `matplotlib` / `pygame` / `fastapi` / `uvicorn` / `websockets`。

##### 可用模式

```bash
uv run python enhancedSnake.py play                  # 观看已有模型（默认启用安全规划）
uv run python enhancedSnake.py train 2000            # 训练 2000 局
uv run python enhancedSnake.py play --unsafe         # 仅对比：关闭安全层，观察原始 DQN
uv run python enhancedSnake.py manual                # 方向键控制，空格重开
uv run python enhancedSnake.py perfect 240           # 哈密顿环安全策略（参数为显示帧率）
uv run python enhancedSnake.py evaluate --games 20   # 无窗口对比四种策略，输出 JSON
uv run python webapp.py                              # Web 控制台 → http://127.0.0.1:8333
```

Web 控制台在同一界面里提供全部四种模式：实时棋盘、键盘与屏幕方向键、运行速度与智能规划开关、
训练曲线监控。

##### 关键设计

- **默认 AI = 模型 + 安全层 + BFS 规划**：先过滤「可能追上尾巴或沿环越过食物」的动作，
  再在保持蛇身环形顺序的空闲区域里搜索最短路线，DQN 在同样短的候选路线中决策；食物被身体挡住时
  沿环推进最多的安全动作，等蛇尾释放空间。**每步重新规划，不做易过期的路径缓存**。
- **DQN 现代化**：训练与推理都排除反向无效动作；新训练使用 Double DQN（在线网络选动作、
  目标网络估值）与 Huber 损失，抑制价值高估与异常奖励带来的梯度波动。网络仍为 `10→64→64→4`。
- **训练环境默认 500 步后终止**，避免经验回放被死循环占据；手动模式与完美策略不受此限制。
- `--unsafe` 关闭全局安全约束和规划，只保留模型与单步合法动作掩码。

##### 评测结果

使用仓库已有模型、默认初始蛇身、食物种子 0–19，每局 500 步：

| 策略 | 平均得分 | 碰撞局数 |
| --- | ---: | ---: |
| 原模型 + 原哈密顿安全层（`shield`） | 17.80 | 0 / 20 |
| 原模型 + 安全最短路径规划（`smart`，默认） | 22.25 | 0 / 20 |

默认策略平均得分提高 **25%**。完美策略长局对比（相同初始蛇身、种子 0–4）优化前后均为
5/5 胜利，铺满棋盘平均步数由 47,789.6 降至 37,597.6（约 **-21.3%**）。以上均为固定种子样本结果，
不代表所有种子的提升幅度。

##### 测试

```bash
cd snake-ai
uv run python -m unittest discover -s tests -v
```

覆盖环境边界、模型、训练目标、路径规划与 Web 游戏回归。

---

#### 项目四：python_homework —— 雷达脉内调制样式识别

单脉冲识别流程：复数 IQ 信号 → STFT 功率谱 → STFT-CNN → 调制类别。附一个中文 Web 工作台，
支持样例生成、IQ 上传、CPU 预测、时频图 / 波形展示、实验评估与 JSON 结果导出。
详细说明见 `README.md`。

##### 与开源项目的关系

**没有重写 CNN**，而是直接导入上游
[seb-cocks/AIMC-image-classification](https://github.com/seb-cocks/AIMC-image-classification)
的两个模块，放在 `third_party/`（固定提交 `6d50291`，原始源码保留）：

- `src/STFT_CNN/STFT_CNN.py` 的 `MainModel`
- `src/common/spectrogram.py` 的 `batched_iq_to_power_spectrogram`

##### 两组实验（结论请以「小样本演示」看待）

| 实验 | 数据 | 配置 | 结果 |
| --- | --- | --- | --- |
| 最初的 8 类**合成**演示 | 自行生成的 8 类单脉冲 IQ，训练 2048 / 验证 512 / 测试 800 | 1024 点、1 MHz、96×96 输入，30 轮 CPU 训练 | 独立测试准确率 **71.00%**，Macro-F1 **0.7058**；16 条新信号中 13 条正确 |
| 官方 30 类**简化**版 | AIMC-Spec-v2 官方测试文件，每类 100 训练 / 25 验证 / 50 测试（按原始脉冲划分） | 1000 点、50 MHz，6 次方差预处理，20 轮 CPU 训练 | 7500 条留出测试准确率 **7.61%**，Macro-F1 **0.0654**；30 条 6 dB 留出信号仅 2 条正确 |

需要强调的是：官方 30 类实验是**用官方信号做的小规模演示，不是官方完整训练集实验，
也不是官方固定测试协议的基准结果**；该权重只适合展示流程，不能当作可靠的调制识别结果。
最初的 8 类实验数据为自造合成信号，同样不是论文指标复现。

8 类调制样式：`CW` / `LFM_UP` / `LFM_DOWN` / `NLFM` / `BPSK` / `BFSK` / `4FSK` / `SFM`。
SNR 为 `[-6, -3, 0, 3, 6] dB`。

##### 运行

> 该项目的命令按 POSIX（Linux 开发环境）书写。现有 `.venv` 已装依赖，`-E` 用于避免本机
> ROS 的 `PYTHONPATH` 干扰。

```bash
cd radar-iq

# 一条命令：重新生成数据 → 训练 → 测试 → 喂入新信号
.venv/bin/python -E -m aimc demo --epochs 30 --device cpu

# 中文 Web 工作台 → http://127.0.0.1:8080
python3 -B -E -m aimc.web --port 8080

# 官方 30 类简化版（默认单线程、计算占空比 25%、nice 10、不使用 GPU）
nice -n 10 .venv/bin/python -E -m aimc official-demo --epochs 20

# 分步执行
.venv/bin/python -E -m aimc generate
.venv/bin/python -E -m aimc train --epochs 30 --device cpu
.venv/bin/python -E -m aimc evaluate --input data/synthetic/test.npz
.venv/bin/python -E -m aimc predict --input your_iq.npy --sample-rate 1000000 --min-confidence 0.7
```

支持的输入格式：`.npy`（复数 `(N,)`/`(B,N)`，或实数 `(2,N)`/`(N,2)`/`(B,2,N)`/`(B,N,2)`）、
`.npz`（`iq` 字段，可附 `sample_rate_hz`）、无表头两列 `I,Q` 的 `.csv`。
程序会拒绝 NaN / Inf / 全零脉冲 / 长度不匹配 / 采样率不匹配的数据，**不自动裁剪、补零或重采样**。

Web 工作台上传每次最多 64 条脉冲、8 MB；识别记录只保存预测元数据，**不保存上传的原始 IQ**。

##### 依赖与测试

```bash
.venv/bin/python -m pip install -r requirements.txt   # numpy / torch / matplotlib / h5py / Flask
.venv/bin/python -E -m pytest -q tests
```

`requirements-lock.txt` 记录本次 Python 3.13 CPU 环境的具体版本；GPU 需自行安装适配驱动的
PyTorch 后使用 `--device cuda`。`data/` 与 `outputs/`（原始数据与训练产物）**不纳入版本管理**。

---

#### 项目五：file-finder —— ffind 文件查找器

以「快」为第一目标的文件名查找器：**一次并行扫描**建立文件名索引，之后所有查询只碰一个 SQLite
文件，毫秒级返回。

> 与 `find` 的根本区别：`find` **每次查询**都要从根目录重新遍历整棵目录树；
> `ffind` 把遍历成本摊到**一次性建索引**上，查询时不再碰磁盘目录结构。树越大、查得越勤，
> 优势越明显。

##### 它为什么快

- **扫描阶段**：用 `os.scandir()` 一次 `readdir` 同时拿到文件名与类型（Linux `d_type`），
  **不对每个条目 `stat`**；多进程并行 + **动态任务切分**（worker 扫够一批就提前交还，负载自然均衡）；
  进程间用 NUL 拼接的单个大字符串传路径，避开 pickle 传大量小字符串的开销。
- **索引阶段**：`WITHOUT ROWID` 表以 `path` 为主键，**先灌数据、最后一次性 `CREATE INDEX`**
  （把逐行维护 B-tree 退化成一次外部排序，实测快 2.3×）；全程单事务 + `journal_mode=OFF`。
- **查询阶段**：前缀查询直接走 `(name, path)` 覆盖索引做范围扫描，不回表、不排序；
  连接开 256MB mmap 读索引，省掉逐页 `read()`。

##### 环境准备与运行

**纯标准库，无第三方依赖（Python ≥ 3.10）**。三种方式任选：

```bash
cd file-search

# A. 用 uv（推荐，免安装直接跑）
uv run --no-project python -m ffinder <关键词>

# B. 装成命令行工具（得到 `ffind` 命令）
uv pip install -e .      # 或 pip install -e .
ffind <关键词>

# C. 任何目录下直接跑
PYTHONPATH=/path/to/file-finder python3 -m ffinder <关键词>
```

##### 常用用法

```bash
ffind update ~/code /data          # 建索引（默认家目录 ~）；可给多个根
ffind report pdf                   # 文件名同时包含 report 和 pdf（AND）
ffind -e md -e rst readme          # 扩展名为 md/rst 且含 readme
ffind -t dir config                # 只找目录
ffind -r '^test_.*\.py$'           # 正则（匹配文件名）
ffind -g '*.cc' -p src             # 通配符，且限定在 ./src 子树下
ffind -l -e py main                # 附带大小与修改时间
ffind -0 foo | xargs -0 rm         # NUL 分隔，安全管道
ffind --fresh 30 notes             # 索引超 30 分钟则先自动重建再搜
ffind status                       # 查看索引规模 / 时间 / 根
```

常用选项：`-r/--regex`、`-g/--glob`、`-e/--ext`、`-t/--type`、`-p/--from`、
`-x/--exclude`、`-i/--case-sensitive`、`-n/--limit`、`-c`/`-l`/`-0`、
`--rebuild`/`--fresh MIN`、`-j/--jobs`、`--db PATH`。首次搜索若无索引会自动建一次。

##### Web 界面

```bash
ffind serve --open                 # 启动并自动打开浏览器（默认 http://127.0.0.1:8765）
ffind serve --port 9000            # 换端口
ffind serve -p ~/code -p /data     # 限定可搜索 / 打开的根目录
```

实时搜索（110ms 防抖 + 取消过期请求，空格分隔多词 AND，命中高亮）、一键后台重建并实时显示进度
（期间旧索引仍可搜，完成后原子替换）、键盘操作（`/` 聚焦、`↑`/`↓` 选择、`Enter` 打开、
`Esc` 清空）、**零依赖**（标准库 `http.server` + 单文件内联前端）。服务**只监听 `127.0.0.1`**，
不对外网暴露。

##### 实测性能

机器：Linux 6.8 / 32 核 / NVMe SSD，缓存预热（与 `find` 同条件对比）：

```
$ python bench.py --files 200000            # 20 万文件的合成树
find 全量遍历:              143 ms
find 子串过滤(*pat*):       122 ms / 次查询
ffind 建索引(一次性):       451 ms          (454k 条目/秒)
ffind 子串查询:             14 ms / 次      → 比 find 快 ~9×
ffind 前缀查询(走索引):      2.6 ms / 次     → 比 find 快 ~48×

$ ffind update /usr --db /tmp/x.db          # 真实 /usr，38.4 万条目
建索引: 791 ms   (348,241 文件 / 36,177 目录, 486k 条目/秒)
前缀 lib : 1.7 ms  vs find 460 ms  → ~270×
子串 conf: 1.9 ms  vs find 450 ms  → ~235×
```

##### 设计边界（诚实说明）

- **只索引文件名，不搜内容**（要按内容搜请用 `grep`/`ripgrep`，二者正交）；
- 索引是**快照**，新建/改名/删除的文件需要 `--rebuild` 或 `--fresh` 才反映（与 `locate` 一致）；
- 正则 / 通配（`-r`/`-g`）无 `LIKE` 可下推，会扫全索引再用 Python 过滤，比关键字查询慢一到两个数量级；
- 大小写不敏感按 **ASCII**（SQLite `NOCASE`），中文文件名子串匹配正常；
- 默认不跟随符号链接，默认排除 `.git`、`node_modules`、`.cache` 等目录名；
- **重建不打断使用**：先写 `index.sqlite.tmp`，完成后 `os.replace` 原子替换，中途中断旧索引完好；
- 单次查询最多排序 2 万条候选（`_CAP`）防极端模式吃爆内存。

##### 测试

```bash
uv run --no-project --with pytest python -m pytest tests -q
```

---

#### 环境与依赖汇总

| 项目 | Python | 第三方依赖 | 其他要求 |
| --- | --- | --- | --- |
| SimpleChat | 3.x（标准库即可） | **无** | 无需前端构建，静态文件由后端一并托管 |
| SimplePlayer | 需带 `tkinter` | `ffpyplayer==4.5.3`、`tkinterdnd2==0.6.3` | 已内嵌 FFmpeg，无需单独安装 |
| snake | 3.13 | torch(CPU)、numpy、matplotlib、pygame、fastapi、uvicorn、websockets | 用 `uv sync` 管理；仓库内已有可用权重 |
| python_homework | 3.10+ | torch、numpy、matplotlib、h5py、Flask（GPU 需自行装 CUDA 版 torch） | 上游源码在仓库里只是子模块指针，新克隆需手动补；代码含 Unix 专用调用，Windows 需 WSL 或补丁 |
| file-finder | ≥ 3.10 | **无**（纯标准库） | 可选 `pyproject.toml` 安装以获取 `ffind` 命令 |

#### 备注

- `.venv/`、`data/`、`outputs/`、`__pycache__/` 等已由 `.gitignore` 排除；克隆后请按各项目
  文档重新创建虚拟环境并安装依赖。仓库根目录 `.python-version` 指向 3.13。
- SimpleChat 已按「单端口 + 反向代理」的形态设计，可直接部署到支持 `$PORT` 环境变量的托管环境
  （启动命令 `python server.py`）。
- 每个项目目录下的 `README.md` 含该项目的完整说明、参数与踩坑记录，本文档只做总览。
- **实际运行**（依赖安装、启动命令、端口、常见报错）见根目录 `运行指南.md`。



---

<a id="source-readme-1"></a>
#### 附录 B · file-search/README.md

以下命令和相对文件名以 `file-search/` 为工作目录。

### ffind —— 以「快」为第一目标的文件查找器

一次**并行扫描**建立文件名索引,之后所有查询只碰一个 SQLite 文件,
毫秒级返回。适合反复找文件、记不全名字、只想按关键字/类型/路径过滤的场景。

> 与 `find` 的根本区别:`find` **每次查询**都要从根目录重新遍历整棵目录树;
> `ffind` 把遍历成本摊到**一次性建索引**上,查询时不再碰磁盘目录结构。
> 树越大、查得越勤,优势越明显。

#### 它为什么快

**扫描阶段(`ffinder/scan.py`)**

1. `os.scandir()` 一次 `readdir` 同时拿到文件名和类型(Linux `d_type`),
   **不对每个条目 `stat`** —— 这正是 `os.walk`/`glob` 最大的隐藏开销。
2. **多进程并行** + **动态任务切分**:worker 扫够一批就提前返回,把没扫完的
   子目录交还父进程重新分发。大目录(如 `node_modules`)和小目录天然负载均衡。
3. 进程间用 **NUL 拼接的单个大字符串**传路径,避开 pickle 传 20 万个小字符串对象的开销。

**索引阶段(`ffinder/index.py`)**
4. `WITHOUT ROWID` 表以 `path` 为主键;**先灌数据、最后一次性 `CREATE INDEX`**,
   把「每插一行维护一次 B-tree」退化成一次外部排序(实测建索引快 2.3×)。
5. 建索引全程单事务、`journal_mode=OFF`、`synchronous=OFF`。

**查询阶段**
6. **前缀查询**(`ffind lib`)直接走 `(name, path)` 覆盖索引做范围扫描,
   **不回表、不排序**,几乎瞬时。
7. **子串查询**(`ffind conf`)命中不足时才补一次覆盖索引全扫,全程不读真实文件。
8. 查询连接开 **256MB mmap** 读索引,省掉逐页 `read()` 系统调用。

#### 安装 / 运行

纯标准库,无第三方依赖(Python ≥ 3.10)。三种方式任选:

```bash
# A. 用 uv(推荐,免安装直接跑)
uv run --no-project python -m ffinder <关键词>

# B. 装成命令行工具(得到 `ffind` 命令)
uv pip install -e .      # 或 pip install -e .
ffind <关键词>

# C. 任何目录下直接跑
PYTHONPATH=/home/time/python_2026/file-finder python3 -m ffinder <关键词>
```

#### 用法

```bash
ffind update ~/code /data          # 建索引(默认家目录 ~);可给多个根
ffind report pdf                   # 文件名同时包含 report 和 pdf(AND)
ffind -e md -e rst readme          # 扩展名为 md/rst 且含 readme
ffind -t dir config                # 只找目录
ffind -r '^test_.*\.py$'           # 正则(匹配文件名)
ffind -g '*.cc' -p src             # 通配符,且限定在 ./src 子树下
ffind -p /usr lib conf             # 限定路径下,多关键词 AND
ffind --fresh 30 notes             # 索引超 30 分钟则先自动重建再搜
ffind -c item                      # 只要命中数量
ffind -l -e py main                # 附带大小与修改时间
ffind -0 foo | xargs -0 rm         # NUL 分隔,安全管道
ffind status                       # 查看索引规模/时间/根
ffind -- update                    # 要搜文件名恰好叫 update/status 时,用 -- 转义
```

首次搜索若无索引会**自动建一次**(带 `--from` 时只索引该路径,不会擅自动整个家目录)。

**常用选项**

| 选项                           | 含义                                       |
| ---------------------------- | ---------------------------------------- |
| `-r/--regex` `-g/--glob`     | 正则 / 通配符匹配文件名(需扫全表,较慢)                   |
| `-e/--ext`                   | 限定扩展名,可重复                                |
| `-t/--type` `file\|dir\|any` | 结果类型(默认 any)                             |
| `-p/--from PATH`             | 限定搜索根,可重复                                |
| `-x/--exclude NAME`          | 查询期按目录名过滤,可重复                            |
| `-i/--case-sensitive`        | 区分大小写(默认按 ASCII 不敏感)                     |
| `-n/--limit`                 | 最多显示条数,`0`=不限(默认 200)                    |
| `-c` `-l` `-0`               | 只计数 / 长格式 / NUL 分隔                       |
| `--rebuild` `--fresh MIN`    | 强制重建 / 过期自动重建                            |
| `-j/--jobs`                  | 建索引并行进程数(默认自适应)                          |
| `--db PATH`                  | 索引文件位置(默认 `~/.cache/ffind/index.sqlite`) |

#### Web 界面

不想敲命令行?一条命令起一个本地网页,浏览器里实时搜索、可视化重建索引:

```bash
ffind serve --open            # 启动并自动打开浏览器(默认 http://127.0.0.1:8765)
ffind serve --port 9000       # 换端口
ffind serve -p ~/code -p /data  # 限定可搜索/打开的根目录
```

特性:

- **实时搜索**:输入即搜(110ms 防抖 + 取消过期请求),空格分隔多词 AND,
  命中关键词高亮,显示文件名/父路径/大小/修改时间,右上角报告命中数与查询耗时。
- **一键重建**:面板里改搜索根、追加排除目录,后台并行重建并**实时显示进度**
  (已收录 N 文件 / M 目录 · 秒数),期间旧索引仍可搜,完成后原子替换。
- **删除索引**:右上角一键删掉 `index.sqlite`(含残留 `.tmp`/`-wal`/`-journal`),
  **不动任何真实文件**,重建前会弹确认并列出将要释放的文件数/条目/体积;
  删除后界面回到「尚无索引」,搜索根设置保留,可随时重建。重建进行中该按钮不可用(接口返回 409)。
- **结果操作**:单击选中，双击或按 `Enter` 打开目录：文件夹打开自身，文件只打开其所在目录，不打开文件本身。Windows 使用资源管理器，macOS / Linux 使用系统文件管理器。
- **键盘操作**:`/` 聚焦搜索,`↑`/`↓` 选择,`⧉` 复制路径,`Esc` 清空。
- **零依赖**:Python 标准库 `http.server` + 单个 `web/index.html`(内联 CSS/JS),
  **只监听 `127.0.0.1`**,不对外网暴露。打开目录受限于索引根范围,越界返回 403。

> WSL/远程开发:VS Code 会自动转发 `127.0.0.1:8765`,直接在本机浏览器打开即可。

#### 实测性能

机器:Linux 6.8 / 32 核 / NVMe SSD,缓存预热。**热缓存下对 `find` 单次查询的对比**(双方同条件)。

```
$ python bench.py --files 200000          # 20 万文件的合成树
find 全量遍历:              143 ms
find 子串过滤(*pat*):       122 ms / 次查询
ffind 建索引(一次性):       451 ms        (454k 条目/秒)
ffind 子串查询:             14 ms / 次    → 比 find 快 ~9×
ffind 前缀查询(走索引):      2.6 ms / 次   → 比 find 快 ~48×
```

```
$ ffind update /usr --db /tmp/x.db         # 真实世界 /usr,38.4 万条目
建索引: 348,241 文件 / 36,177 目录, 791 ms   (486k 条目/秒)
前缀 lib  :  1.7 ms   vs  find 460 ms   → ~270×
子串 conf :  1.9 ms   vs  find 450 ms   → ~235×
```

结论:**建索引是一次性成本**(每 10 万条目约 0.2 秒,自适应核数),
之后每次查询比 `find` 快一到两个数量级,且树越大、查询越频繁越划算。

#### 测试

```bash
uv run --no-project --with pytest python -m pytest tests -q
```

覆盖:扫描器的不 stat/提前返回分批/嵌套根去重,查询的前缀优先/多词 AND/
大小写/正则/通配/扩展名/类型/`-p` 路径限定/`-x` 过滤/`LIKE` 元字符转义,
以及 CLI 冒烟和增量重建。

#### 设计边界(诚实说明)

- **文件名索引,不搜内容**。要按内容搜请用 `grep`/`ripgrep`;二者正交。
- 索引是**快照**:新建/改名/删除的文件要 `--rebuild` 或 `--fresh` 才反映。
  这与 `locate` 一致 —— 用「新鲜度」换「查询速度」。
- **正则/通配**(`-r`/`-g`)无 `LIKE` 可下推,会扫全索引再用 Python 过滤,
  比关键字查询慢一到两个数量级(实测 /usr 上 167 ms)。
- 大小写不敏感按 **ASCII**(SQLite `NOCASE`)。中文文件名子串匹配正常,
  仅「大小写折叠」只覆盖 ASCII。
- 默认不跟随符号链接(与 `locate`/`plocate` 一致);默认排除 `.git`、
  `node_modules`、`.cache` 等目录名,`update -x` 可追加、
  `update --no-default-excludes` 可整体关闭。
- **重建不打断使用**:建索引先写 `index.sqlite.tmp`,全程不碰正式文件,
  完成后 `os.replace` 原子替换。中途 Ctrl-C / 断电,旧索引完好可查;
  残留的 `.tmp` 会在下次构建时被自动复用重建。
- 单次查询最多排序 2 万条候选(`_CAP`)防极端模式吃爆内存,超出会标注 `+`。
- Web 界面**只监听 `127.0.0.1`**,不对局域网/公网暴露；结果只打开文件夹或文件所在目录
  受限于索引根目录范围,越界路径返回 403。它是本地工具的可视化外壳,不是多用户服务。

#### 对比同类

| 工具                 | 机制       | 查询是否需预建索引 | 语言           |
| ------------------ | -------- | --------- | ------------ |
| `find`             | 每次全量遍历   | 否         | C            |
| `fd`               | 每次并行遍历   | 否         | Rust         |
| `locate`/`plocate` | 预建索引,查索引 | 是         | C            |
| **`ffind`**        | 预建索引,查索引 | 是         | Python(纯标准库) |

`ffind` 的定位:`plocate` 的「索引换速度」思路,做成一个**零依赖、可嵌入、
扫描引擎自身也快**的 Python 库 + CLI + 本地 Web 界面。

#### 目录结构

```
file-search/
├── ffinder/
│   ├── scan.py           # 并行 scandir 扫描引擎(动态任务切分)
│   ├── index.py          # SQLite 覆盖索引:建库 + 查询 + 排序
│   ├── cli.py            # 命令行(update / status / serve / 搜索)
│   ├── server.py         # 零依赖 Web 后端(http.server,仅 127.0.0.1)
│   ├── web/index.html    # 单文件前端(内联 CSS/JS,深浅色主题)
│   └── __main__.py       # python -m ffinder 入口
├── tests/
│   ├── test_ffind.py     # 扫描/索引/查询 + CLI 正确性
│   └── test_server.py    # Web API 端点(起真实 HTTP 服务)
├── bench.py              # 对 find 的速度基准
└── pyproject.toml        # 可选安装,暴露 ffind 命令
```



---

<a id="source-readme-2"></a>
#### 附录 C · radar-iq/README.md

以下命令和相对文件名以 `radar-iq/` 为工作目录。

### 雷达脉内调制样式识别

Python 实现的单脉冲识别流程：复数 IQ 信号 → STFT 功率谱 → STFT-CNN → 调制类别。
已在当前目录创建 `.venv`，实际完成 CPU 训练、独立测试和新信号推理。

#### 本地 Web 工作台

安装 `requirements.txt` 后启动中文界面：

```bash
python3 -B -E -m aimc.web --port 8080
```

浏览器打开 `http://127.0.0.1:8080`。工作台提供样例信号生成、IQ 文件上传、
CPU 模型预测、功率时频图、IQ 波形、实验评估、模型参数和 JSON 结果导出。
上传支持 `.npy`、`.npz` 和无表头两列 `I,Q` 的 `.csv`，每次最多 64 条脉冲、8 MB。
合成模型要求 1024 点 / 1 MHz，官方演示模型要求 1000 点 / 50 MHz。
识别记录只在当前浏览器保存预测元数据，不保存上传的原始 IQ。
界面使用已有 `outputs/` 权重和实验结果，不触发训练或数据下载。
若复制项目后 `.venv` 的 Python 链接失效，需使用可用的 Python 环境或重建虚拟环境。

#### 最新简化版：官方 30 类数据

官方数据可以下载。已获取并校验新版 AIMC-Spec 的全部 30 个测试类别文件，
每类 1000 条原始脉冲。数据来源是作者公开发布的
[AIMC-Spec-v2](https://www.kaggle.com/datasets/sebastiancocks/aimc-spec-v2)，版本 1。
[新版论文](https://arxiv.org/abs/2601.08265v2)明确写 30 类，原仓库 README 的 33 类描述与之不一致。

按用户要求采用简单、低负载方案：停止完整训练集下载和 CUDA 安装，保留现有 CPU 环境。
从已下载的官方测试文件中，每类选 100 条训练、25 条验证、50 条测试，按原始脉冲重新划分。
排除全零、非有限值和重复原始波形，每个脉冲的 5 档 SNR 样本全部保留在同一集合。
加噪后分别为 15000、3750、7500 条，覆盖全部 30 类。
这是使用官方信号的小规模演示，**不是官方完整训练集实验，也不是官方固定测试协议的基准结果**。

运行默认 20 轮训练并喂入留出脉冲：

```bash
nice -n 10 .venv/bin/python -E -m aimc official-demo --epochs 20
```

该命令默认仅使用 1 个 CPU 线程，计算占空比 25%，进程优先级 nice 10，不使用 GPU。
官方简化版使用原 STFT-CNN 的 6 次方差预处理，学习率为 0.0006，输入图像为 96×96。
`outputs/official_demo/resources.json` 和 `resources.jsonl` 记录计算时间、休息时间和内存占用。
占空比限制的是本训练进程的计算/休息时间，不是整台电脑的固定利用率。
原始数据在 `data/official/raw/AIMC_Spec_v2_test/`，划分及排除清单在 `data/official_demo/manifest.json`。
解码优先使用 `hdf5plugin`；本机未安装该插件时，通过 HDF5 原始块接口调用已有 `libblosc`，没有重新合成官方干净信号。

训练权重、测试指标、曲线、混淆矩阵及留出脉冲预测写入 `outputs/official_demo/`。
本次已实际完成 20 轮：7500 条留出测试样本准确率 **7.61%**，Macro-F1 **0.0654**；
喂入每类一条 6 dB 留出信号，30 条中仅 **2 条正确**。
训练约 18 分钟，计算占空比约 24.6%，包括测试在内的进程峰值内存约 1.9 GB。
这个小样本模型的识别效果很差，权重仅适合展示流程，不能当作可靠的调制识别结果。
该模型的 IQ 窗口是 **1000 点、50 MHz 采样率**，与下文早期 8 类模型的配置不同。

```bash
.venv/bin/python -E -m aimc predict --checkpoint outputs/official_demo/best.pt --input data/official_demo/heldout_pulses.npz --threads 1
```

安装依赖使用 `requirements.txt`；若系统无 `libblosc`，另安装 `hdf5plugin`。
未下载完成的压缩包及已经解压的少量官方训练文件保留在 `data/official/`，当前没有下载或训练后台任务。

#### 最初的 8 类合成演示

以下说明记录最初实现的 8 类实验；其数据、权重和结果保留不变。

#### 与开源项目的关系

直接导入 [seb-cocks/AIMC-image-classification](https://github.com/seb-cocks/AIMC-image-classification) 的两个源码模块，没有重写其 CNN：

- `src/STFT_CNN/STFT_CNN.py` 中的 `MainModel`。
- `src/common/spectrogram.py` 中的 `batched_iq_to_power_spectrogram`。

上游放在 `third_party/AIMC-image-classification/`，固定提交为
`6d50291fd3ad29389dff872b3ce73c06fb6f5c7c`，原始源码保留。

最初演示的数据是自行生成的 8 类单脉冲 IQ，**不是官方 AIMC-Spec 数据，也不是论文指标复现**。
采用原函数支持的每时刻峰值参考、40 dB 动态范围归一化，双线性缩放到 96×96，灰度重复为三通道。
通过原模型参数 `use_paper_preprocess=False` 关闭迭代绝对值/方差预处理，CNN 主干保持原样。
原项目导出器默认使用 224×224 和另一种噪声归一化，本地配置与其有所区别。
任务为单标签分类，训练使用交叉熵；不处理多发射源重叠的多标签识别。

#### 类别与信号

| 标签         | 调制样式       |
| ---------- | ---------- |
| `CW`       | 未调制单频脉冲    |
| `LFM_UP`   | 线性上扫频      |
| `LFM_DOWN` | 线性下扫频      |
| `NLFM`     | 幂函数型非线性上扫频 |
| `BPSK`     | 二相编码       |
| `BFSK`     | 二频移键控      |
| `4FSK`     | 四频移键控      |
| `SFM`      | 正弦调频       |

默认每条 IQ 有 1024 点，采样率为 1 MHz。载频、带宽、初相、幅度、码元数等随机变化；
SNR 为 `[-6, -3, 0, 3, 6] dB`，定义为脉冲平均信号功率与复高斯噪声期望功率之比。
训练、验证、测试分别使用随机种子 42、43、44，独立生成。训练前还会检查跨集合完全重复的 IQ。
这些是简化信号模型，不覆盖实际接收机失真、未知类别和完整工程信号条件。

#### 已完成的训练

| 项目            | 本次运行                                |
| ------------- | ----------------------------------- |
| 环境            | Python 3.13，PyTorch CPU，4 线程        |
| 数据量           | 训练 2048，验证 512，测试 800               |
| 训练            | 30 轮，batch 64，Adam，初始学习率 0.003，余弦衰减 |
| 最佳权重          | 按验证损失选取第 27 轮                       |
| 独立测试准确率       | **71.00%**                          |
| 独立测试 Macro-F1 | **0.7058**                          |
| 新生成信号         | 6 dB，独立种子 98765，16 条中 **13 条正确**    |

测试集没有用于早停或选择权重。上述数值仅针对本地合成分布。
当前模型对 CW/BPSK、BFSK/4FSK 的区分仍较弱，具体见分类指标和混淆矩阵。
softmax 分数未做置信度校准，高分也可能预测错误；`--min-confidence` 仅用于标记低分结果，不能可靠检测未知调制。

结果文件：

- `outputs/stft_cnn/best.pt`：权重、类别顺序及预处理配置。
- `outputs/stft_cnn/config.json`：训练配置及上游版本。
- `outputs/stft_cnn/history.json`、`history.csv`：每轮训练/验证记录。
- `outputs/stft_cnn/metrics.json`：测试准确率、Macro-F1、分类别和分 SNR 指标。
- `outputs/stft_cnn/fresh_predictions.json`：16 条新信号的预测、真实标签和分数。
- `outputs/stft_cnn/single_prediction.json`：单条新信号的预测结果。

![训练曲线](outputs/stft_cnn/training_curves.png)
![混淆矩阵](outputs/stft_cnn/confusion_matrix.png)
![各类功率时频图](outputs/stft_cnn/spectrograms.png)

#### 运行

在当前目录执行。现有 `.venv` 已安装依赖，`-E` 可避免本机 ROS 的 `PYTHONPATH` 干扰。

一条命令重新生成数据、训练、测试、喂入新信号，会更新默认数据和结果目录：

```bash
.venv/bin/python -E -m aimc demo --epochs 30 --device cpu
```

分步执行：

```bash
.venv/bin/python -E -m aimc generate
.venv/bin/python -E -m aimc train --epochs 30 --device cpu
.venv/bin/python -E -m aimc evaluate --input data/synthetic/test.npz
```

生成并喂入一条新信号：

```bash
.venv/bin/python -E -m aimc sample --class LFM_UP --snr 6 --output data/new_pulse.npy
.venv/bin/python -E -m aimc predict --input data/new_pulse.npy --sample-rate 1000000 --output outputs/stft_cnn/single_prediction.json
```

喂入自己的 IQ：

```bash
.venv/bin/python -E -m aimc predict --input your_iq.npy --sample-rate 1000000 --min-confidence 0.7
```

支持的文件格式：

- `.npy`：复数 `(N,)` 或 `(B,N)`；实数 `(2,N)`、`(N,2)`、`(B,2,N)` 或 `(B,N,2)`。
- `.npz`：`iq` 字段使用上述形状；可附带标量 `sample_rate_hz`，程序会核对采样率。
- `.csv`：无表头，每行两列 `I,Q`，表示一个脉冲。

默认权重要求每个脉冲恰好 1024 点，训练和推理应采用相同的采样率及脉冲分割方式。
程序会拒绝 NaN、Inf、全零脉冲、长度不匹配和已知采样率不匹配的数据。
没有采样率元数据的文件需调用方保证采样率正确。这里不自动裁剪、补零或重采样。

#### 用自己的标注数据训练

准备 `data/real/train.npz`、`val.npz`、`test.npz`。三个集合须独立、类别顺序一致。
实际数据应按采集批次/发射源划分，避免同一脉冲或同源近重复样本跨集合。
每个文件的字段如下：

```python
import numpy as np

# iq: complex64，形状 (B,1024)，或上述实数 IQ 形状。
# labels: int64，形状 (B,)，类别索引从 0 开始。
# classes: 字符串数组，形状 (C,)，顺序就是类别索引顺序。
np.savez_compressed(
    "data/real/train.npz",
    iq=iq,
    labels=labels,
    classes=np.array(["CW", "LFM_UP", "LFM_DOWN", "NLFM", "BPSK", "BFSK", "4FSK", "SFM"]),
    sample_rate_hz=1_000_000,
    # snr_db=snr_db,  # 可选，每条样本一个值，用于分 SNR 评估。
)
```

```bash
.venv/bin/python -E -m aimc train --data-dir data/real --out-dir outputs/real --epochs 50
.venv/bin/python -E -m aimc predict --checkpoint outputs/real/best.pt --input your_iq.npy --sample-rate 1000000
```

自己的训练数据可以使用任意不少于两类的类别名称；合成器限定为上表 8 类。
采样率、脉冲长度可分别用 `train --sample-rate`、`--samples` 配置，推理自动读取权重内的预处理参数。
完整 AIMC-Spec 的 HDF5/多标签 memmap 不能直接当作本地 NPZ 使用；其导出和基准训练见上游 README。

#### 环境与测试

新环境推荐 Python 3.10+：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install pytest
.venv/bin/python -E -m pytest -q tests
```

`requirements-lock.txt` 记录本次 Python 3.13 CPU 环境的具体版本。
GPU 需自行安装适配驱动的 PyTorch，再使用 `--device cuda`；本机此次使用 CPU。
测试覆盖独立划分、SNR 定义、IQ 格式校验、谱图批处理一致性、上游模型反向传播、
指标计算、权重重载和文件推理一致性。

#### 来源

上游项目关联论文：Sebastian L. Cocks 等，AIMC-Spec，arXiv:2601.08265。
参考 https://arxiv.org/abs/2601.08265 和原项目 `README.md`、`ref.bib`。
此次下载的上游仓库未包含单独的 LICENSE 文件，保留来源与作者信息，分发或其他使用前应确认原作者许可。



---

<a id="source-readme-3"></a>
#### 附录 D · radar-iq/third_party/AIMC-image-classification/README.md

以下命令和相对文件名以 `radar-iq/third_party/AIMC-image-classification/` 为工作目录。

### AIMC-Spec Dataset (Updated Version)

##### Spectrogram-Based Radar Intra-Pulse Modulation Classification Benchmark

This repository contains the experimental framework used to benchmark **Automatic Intra-Pulse Modulation Classification (AIMC)** models using a **unified spectrogram representation**.

The project evaluates multiple deep learning architectures on the **same spectrogram input pipeline**, allowing fair comparison of model performance.

This repository hosts the updated release of the AIMC-Spec dataset, a synthetic benchmark designed for reproducible evaluation of Automatic Intrapulse Modulation Classification (AIMC) algorithms.

This version introduces several important improvements over the initial AIMC-Spec release, including:

- redesigned storage pipeline reducing dataset size from >500 GB to <30 GB
- refined modulation set and standardized signal definitions
- streamlined dataset generation process for improved usability
- updated benchmark protocol and experimental evaluation

The updated dataset is fully described in the following paper:
AIMC-Spec: An Updated Benchmark Dataset for Automatic Intrapulse Modulation Classification
arXiv: https://arxiv.org/abs/2601.08265

##### Users are strongly encouraged to consult the paper for:
- dataset design methodology
- signal generation details
- SNR definition and preprocessing
- benchmark evaluation protocol
- recommended train/validation/test usage

---

### Research Focus

The goal of this repository is to compare **model architectures** rather than input representations.

All models use the **same power spectrogram representation** generated from IQ radar data.

Architectures currently implemented include:

- **STFT-CNN**
- **LPI-Net**
- **LDC-UNet**
- **CDAE-DCNN**
- **Vision Transformer (ViT)**

Each architecture has its own training module but uses the same dataset pipeline and experiment framework.

---

### Dataset

The framework operates on the **AIMC-Spec dataset**, which contains radar emitter signals with different intra-pulse modulations.

#### Modulation Classes

33 intra-pulse modulation types including:

##### FM Family
- unmodulated
- LFM up/down
- NLFM variants
- multi-LFM
- dual-LFM

##### Phase-Coded
- BPSK
- QPSK
- Barker codes
- P1–P4 polyphase codes

##### Additional Modulations
- triangle
- exponential
- BFSK
- 4FSK
- EQFM
- SFM
- Costas

#### SNR Levels

Experiments typically use:

```
+6, +3, 0, −3, −6 dB
```

These levels simulate progressively more challenging radar environments.

---

### Repository Structure

```
src/
│
├─ main.py
│
├─ build_iq_train_subset.py
├─ export_spectrogram_memmap.py
│
├─ common/
│   ├─ config.py
│   ├─ training.py
│   ├─ evaluation.py
│   ├─ metrics.py
│   ├─ spectrogram.py
│   ├─ memmap.py
│   └─ MultiHotStreamLoader.py
│
├─ STFT_CNN/
├─ LPI_Net/
├─ LDC_Unet/
├─ CDAE_DCNN/
└─ ViT/
```

##### `common/`

Contains shared infrastructure used by all models:

- dataset loading
- spectrogram generation
- training loops
- evaluation utilities
- metrics
- checkpoint management

##### Model Directories

Each model architecture has its own directory containing:

- model definition
- experiment configuration
- training entrypoint

---

### Dataset Preparation Pipeline

The training pipeline operates in **three stages**.

---

#### 1️⃣ Build Training Subset

Creates a smaller training dataset from the full emitter dataset.

```
build_train_subset()
```

Input:

```
AIMC_Spec_v2_train_ALL/
```

Output:

```
AIMC_Spec_v2_train/
```

---

#### 2️⃣ Export Spectrogram Dataset

Converts the HDF5 IQ dataset into **memmap spectrogram files**.

```
export_main()
```

Output directory example:

```
power_specs_memmap/
```

Files produced:

```
power_u8.dat
yend_u8.dat
snr_i16.dat
mod_i16.dat
stream_i32.dat
k_u8.dat
meta.json
```

These files contain spectrogram images and aligned labels for efficient training.

---

#### 3️⃣ Train Models

Each model can then be trained on the exported dataset.

Example models:

```
stft_main()
ldc_main()
lpi_main()
cdae_main()
vit_main()
```

Outputs are written to:

```
out/<model_name>/
checkpoints/<model_name>/
```

---

### Running Experiments

Run the project entrypoint:

```bash
python -m src.main
```

Inside `main.py`, enable the stage you want to run.

Example:

```python
if __name__ == "__main__":

    stft_main()
```

Only **one stage should normally be enabled at a time**.

---

### Training Outputs

Training generates:

```
out/<model_name>/
    results
    confusion matrices
    logs

checkpoints/<model_name>/
    saved model weights
```

---

### Reproducibility

Experiments use:

- fixed spectrogram generation parameters
- deterministic dataset splits
- configurable training seeds
- consistent dataset loader across models

All architectures therefore operate on the **same dataset and input representation**.

---

### Requirements

Python **3.10+ recommended**.

Install dependencies:

```bash
pip install -r requirements.txt
```

Quick environment test:

```bash
python - <<'PY'
print("Env OK")
import torch, numpy, matplotlib, pandas
print("Imports OK")
PY
```

---

### Citation

If you use this codebase or the AIMC-Spec dataset, please cite the associated publication.

@misc{cocks2026,
  title         = {AIMC-Spec: A Benchmark Dataset for Automatic Intrapulse Modulation Classification under Variable Noise Conditions},
  author        = {Sebastian L. Cocks and Salvador Dreo and Brian Ng and Feras Dayoub},
  year          = {2026},
  eprint        = {2601.08265},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  note          = {arXiv:2601.08265v2}
}

---

### Notes

This repository is part of ongoing research on **radar intra-pulse modulation classification** and the development of **AIMC-Spec benchmarking frameworks**.



---

<a id="source-readme-4"></a>
#### 附录 E · snake-ai/README.md

以下命令和相对文件名以 `snake-ai/` 为工作目录。

### 基于强化学习的贪吃蛇

这是一个面向教学的单文件 Double DQN 贪吃蛇项目，同时提供手动游戏、已训练模型演示和哈密顿环必胜策略。默认 AI 将模型与安全路径规划结合，优先走安全的最短食物路线。核心实现位于 `enhancedSnake.py`；配套 PDF 描述原始 DQN 设计，当前算法改进以代码和本 README 为准。

#### 快速开始

项目使用 Python 3.13 和 [uv](https://docs.astral.sh/uv/) 管理依赖：

```bash
uv sync
uv run python enhancedSnake.py play
```

也可以启动浏览器控制台，在同一个界面中使用全部四种模式：

```bash
uv run python webapp.py
```

然后访问 `http://127.0.0.1:8333`。Web 界面支持实时棋盘、键盘与屏幕方向键、运行速度和智能规划开关，以及训练曲线监控。

可用模式：

```bash
# 训练 2000 局；模型和曲线默认保存到项目目录
uv run python enhancedSnake.py train 2000

# 使用不同随机种子和输出位置训练
uv run python enhancedSnake.py train 500 --seed 7 \
  --model artifacts/model.pth --progress artifacts/progress.png

# 观看仓库内已有模型，可调整局数和单局上限
uv run python enhancedSnake.py play --games 5 --max-steps 500

# 仅用于对比：关闭默认安全层，观察原始 DQN 策略
uv run python enhancedSnake.py play --unsafe

# 方向键控制，空格重新开始
uv run python enhancedSnake.py manual

# 哈密顿环安全策略；可选参数为显示帧率
uv run python enhancedSnake.py perfect 240

# 无窗口对比四种策略，使用相同种子和步数上限，输出 JSON
uv run python enhancedSnake.py evaluate --games 20 --seed 0 --max-steps 500

# 长局评测：验证默认 AI 能否最终铺满棋盘
uv run python enhancedSnake.py evaluate --strategy smart --games 5 --max-steps 200000
```

查看某个模式的完整参数：

```bash
uv run python enhancedSnake.py train --help
```

#### 项目结构

- `enhancedSnake.py`：环境、DQN、训练、推理、手动游戏和完美策略。
- `demo_rl_snake.py`：直接播放已有模型的轻量入口。
- `snake_model.pth`：与当前网络结构兼容的已训练权重。
- `training_progress.png`：最近一次训练的得分和损失曲线。
- `基于强化学习的贪吃蛇游戏_详细设计文档.pdf`：教学设计文档。
- `tests/`：环境边界、模型、训练目标、路径规划和 Web 游戏回归测试。

#### 测试

测试不打开游戏窗口：

```bash
uv run python -m unittest discover -s tests -v
```

训练环境默认在 500 步后终止，避免经验回放被死循环占据；手动模式和完美策略不使用这个训练上限。AI 演示默认启用哈密顿安全层与 BFS 路径规划：先过滤可能追上尾巴或沿环越过食物的动作，再在保持蛇身环形顺序的空闲区域中搜索最短路线，DQN 在同样短的候选路线中决策。当食物暂时被身体挡住时，选择沿环推进最多的安全动作，等待蛇尾释放空间。每一步重新规划，不使用容易过期的路径缓存。CLI 和 Web 共用这套逻辑。

`--unsafe` 关闭全局安全约束和规划，保留模型与单步合法动作掩码。训练与推理都会排除反向无效动作；新训练使用 Double DQN（在线网络选动作、目标网络估值）和 Huber 损失，以减少价值高估和异常奖励导致的梯度波动。网络仍为 10→64→64→4，仓库已有权重可直接加载。模型权重和历史训练曲线没有被替换；新训练算法的效果需重新训练后评测。

完美策略允许通过 `--max-steps` 人工限制演示时长，但设置上限后不再承诺本局一定完成。

#### 优化对比

使用仓库已有模型、默认初始蛇身、食物种子 0–19，每局 500 步：

| 策略 | 平均得分 | 碰撞局数 |
|---|---:|---:|
| 原模型 + 原哈密顿安全层（`shield`） | 17.80 | 0 / 20 |
| 原模型 + 安全最短路径规划（`smart`，默认） | 22.25 | 0 / 20 |

默认策略平均得分提高 25%。使用相同初始蛇身和种子 0–4 的完美策略长局对比，优化前后均为 5 / 5 胜利，铺满棋盘平均步数从 47,789.6 降至 37,597.6，减少约 21.3%。这些是固定种子的样本结果，不代表所有种子的提升幅度。

`evaluate` 支持 `raw`（仅模型与单步掩码）、`shield`（原安全层）、`smart`（新规划器）、`perfect`（无模型规划），默认 `all`。报告包含每局种子、得分、步数、终止原因，以及平均得分、每颗食物步数、胜利次数和碰撞次数；步数上限导致的结束单独计为 `timeout`。



---

#### 附录 · 展示页原始使用说明

此部分记录原交互版使用方式；播放器按钮已替换为截图，以本文开头的当前说明为准。

### 课程项目展示网页

双击 **一键启动.bat**：自动启动文件查找器、打开 SimplePlayer 桌面窗口，并在默认浏览器中打开展示网页。服务在后台运行，无需保留命令窗口。再次双击会打开已运行的网页，不会重复启动一组服务。

展示顺序：强化学习贪吃蛇 → 雷达调制识别 → SimpleChat → ffind 文件查找器 → SimplePlayer。四个 Web 项目使用 iframe 原样嵌入，每个都有全屏演示、独立打开和重新加载按钮。Esc 退出全屏；浏览器不支持原生全屏时自动使用铺满窗口的演示模式，也可点击“退出全屏”。

#### 本地运行

- 展示网页默认地址为 http://localhost:8877/，端口被占用时自动尝试至 8897；以启动后打开的地址为准。文件查找器使用自动分配的空闲端口，网页自动连接。
- 文件查找器首次启动自动索引本课程项目目录，数据库保存在本文件夹的 runtime 下。首次索引完成前可能没有搜索结果，可稍后重试。新建或改名后，在原界面重建索引。
- 文件查找器的原始代码在 Windows 上可能不支持“打开结果”（原代码使用 xdg-open/open）；可以使用原界面的复制路径功能。展示网页保留此项目原始实现。
- 播放器是真实 Tkinter 桌面窗口，切换到 SimplePlayer 后拖入本地媒体文件演示；网页按钮也可重新打开播放器。
- 双击 **停止展示.bat** 关闭本组网页与文件查找器服务。播放器独立运行，请关闭它的窗口。关闭浏览器不会停止后台服务。

#### 网络与环境

三个公网项目需要互联网，URL 与分享码完整保留：

- 贪吃蛇：https://snake-ai-console.app.workbuddy.host/?sharecode=7EDSNKWLRL
- 雷达：https://radar-modulation-workbench.app.workbuddy.host/?sharecode=HJIJIXYO66
- 聊天室：https://simplechat-online.app.workbuddy.host/?sharecode=G4RG3DQ4ZB

如果平台修改嵌入策略、要求验证或第三方 Cookie、或临时不可达，请使用对应“独立打开”按钮完成访问，再返回重载；父页面无法绕过公网平台的访问限制。原项目的主题与交互均保留，外围展示页面使用暗色主题。

启动器优先使用 ../media-player/.venv/Scripts/python.exe，备用为 PATH 中的 Python 或 Windows py 启动器。要求 Python ≥ 3.10；展示网页与文件查找器仅使用标准库。播放器需要 Tkinter、ffpyplayer 和可选的 tkinterdnd2，当前仓库现有虚拟环境已验证这些依赖可导入。若缺失，按 media-player/requirements.txt 准备依赖后再启动。

服务只监听 127.0.0.1，不将本地文件服务发布到公网。全部新建的网页、启动器、日志、索引及预览均位于本“汇报”文件夹中；通过相对路径调用原有项目，请保留本文件夹与五个项目的相对位置。

#### 文件说明

- index.html / style.css / app.js：展示页面及交互。
- launch.py：本地服务、播放器启动与重复运行检测。
- 一键启动.bat / 停止展示.bat：Windows 双击入口。
- runtime/：运行时自动生成的索引、状态和日志；不通过网页提供这些文件。
- preview.png：页面验证时保存的预览。

技术说明根据仓库 README 与实际源码编写。雷达模型沿用上游 AIMC-image-classification 的 STFT-CNN，明确区分模型复用与本项目集成工作。未将文档中的操作指令视为额外用户要求。

