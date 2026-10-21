# Python 语言与深度学习 · 课程设计

**小组：Coco**  
**汇报人：林子文，许睿泽**  
**组员：万桦平，吴文涛**

本仓库包含五个独立项目：强化学习贪吃蛇、雷达脉内调制识别、在线聊天室、文件查找器和音视频播放器。本文档统一说明当前目录、运行方式、实现原理和功能边界。

## 目录与项目

| 目录 | 项目 | 主要技术 | 主入口 |
| --- | --- | --- | --- |
| [snake-ai](#snake-ai) | 强化学习贪吃蛇 | PyTorch、Double DQN、BFS、FastAPI、WebSocket | `webapp.py` |
| [radar-iq](#radar-iq) | 雷达脉内调制识别 | NumPy、STFT、PyTorch、CNN、Flask | `python -m aimc.web` |
| [web-chat](#web-chat) | 在线聊天室 | Python 标准库、HTTP 短轮询、原生 JavaScript | `server.py` |
| [file-search](#file-search) | SQLite 文件查找器 | scandir、多进程、SQLite B-tree、HTTP 服务 | `python -m ffinder` |
| [media-player](#media-player) | 音视频播放器 | Tkinter、Canvas、ffpyplayer、FFmpeg | `player.py` / `run.bat` |

```text
Python-深度学习/
├── README.md
├── snake-ai/
├── radar-iq/
├── web-chat/
├── file-search/
├── media-player/
├── 汇报内容/
├── 备份/
└── Python编程语言与深度学习课程设计报告_1.docx
```

`汇报内容/` 目录用于存放课程项目的展示文件。

## 运行环境

| 项目 | Python 与依赖 | 运行环境说明 |
| --- | --- | --- |
| `snake-ai` | Python ≥ 3.13；依赖由 `pyproject.toml` 和 `uv.lock` 管理 | 使用 `uv sync` 准备环境，PyTorch 配置为 CPU 索引 |
| `radar-iq` | Python ≥ 3.10；依赖见 `requirements.txt`，历史版本记录见 `requirements-lock.txt` | 当前资源管理模块使用 `resource` 和 `/proc/meminfo`，本地运行采用 Linux / WSL |
| `web-chat` | Python 3.10+；无第三方依赖 | 使用标准库直接启动 |
| `file-search` | Python ≥ 3.10；无第三方运行依赖 | Windows 可使用 `run.bat`，也可直接运行模块 |
| `media-player` | 带 Tkinter 的 Python；`ffpyplayer==4.5.3`、`tkinterdnd2==0.6.3` | 当前 Windows 环境保留在项目内 `.venv/`，以该环境的 Python 启动 |

下文 PowerShell 命令均以**仓库根目录**为起点，各项目的命令块独立执行。服务运行期间保留对应终端，按 `Ctrl+C` 停止；播放器通过关闭窗口退出。虚拟环境应与目标系统和 Python 版本匹配，复制到新电脑后需重新检查依赖。

<a id="snake-ai"></a>
## snake-ai · 强化学习贪吃蛇

### 功能与入口

提供 AI 演示、手动挑战、人机对决、完美策略和模型训练，并支持运行速度、步数上限及规划路线可视化。Web 前端通过 WebSocket 与 Python 游戏环境通信，桌面模式使用 pygame 绘制棋盘。

| 文件 | 作用 |
| --- | --- |
| `enhancedSnake.py` | 游戏环境、DQN 网络、经验回放、训练、规划与命令行入口 |
| `webapp.py` | FastAPI 服务、WebSocket 游戏循环与训练进度推送 |
| `web/index.html` | 浏览器控制台 |
| `snake_model.pth` | 附带的模型权重 |
| `training_progress.png` | 保留的训练曲线 |
| `pyproject.toml` / `uv.lock` | 依赖配置与锁定记录 |

启动 Web 控制台：

```powershell
Set-Location .\snake-ai
uv sync
uv run python webapp.py
```

默认访问 [http://127.0.0.1:8333](http://127.0.0.1:8333)。可通过 `webapp.py --host ... --port ...` 指定地址；设置环境变量 `PORT` 时，服务使用该端口并默认监听 `0.0.0.0`。

### 决策原理

游戏采用 20 × 20 棋盘。模型输入为 10 维状态：食物相对位置 2 维、四个方向的碰撞危险 4 维、当前朝向独热编码 4 维。网络结构为 **10 → 64 → 64 → 4**，输出上、下、左、右四个动作的 Q 值，表示预计累计折扣奖励。

训练使用奖励反馈、经验回放、在线网络与目标网络。当前实现采用 Double DQN，由在线网络选动作、目标网络估值，并使用 Huber 损失调整预测。演示时加载权重、关闭随机探索，不继续更新网络。

默认运行采用 **DQN + 预判规划器**：

1. 用时间感知 BFS 寻找食物路线，考虑蛇尾逐步让路；模拟吃食后的局面，检查是否仍能通向蛇尾，蛇较长时还检查空白区域连通性。
2. 直接吃食不安全时，选择能通向蛇尾的方向周旋，等待身体释放通道。
3. 蛇较长或长时间未吃到食物时，尝试构建动态哈密顿环；成功后保持沿环顺序并寻找安全捷径。
4. DQN 在规划器保留的候选动作中选择 Q 值最大的方向；只有一个候选时直接执行。每走一格重新决策。

模型只读取局部状态，规划器读取完整棋盘，二者分工不同。Q 值不是成功概率；默认 AI 的表现也不能全部归因于神经网络。

### 常用命令

在 `snake-ai/` 中执行：

```powershell
# 观看附带模型，默认使用预判规划
uv run python enhancedSnake.py play --games 5 --max-steps 500

# 关闭全局规划，保留模型和单步合法动作过滤
uv run python enhancedSnake.py play --unsafe

# 手动模式 / 独立哈密顿策略
uv run python enhancedSnake.py manual
uv run python enhancedSnake.py perfect 120

# 训练到独立输出位置
uv run python enhancedSnake.py train 2000 --model artifacts/model.pth --progress artifacts/training.png

# 固定种子的无窗口策略对照
uv run python enhancedSnake.py evaluate --strategy all --games 20 --seed 0
```

`evaluate` 支持 `raw`、`shield`、`smart`、`lookahead`、`perfect` 和 `all`。`smart` 是哈密顿环捷径规划，`lookahead` 是预判规划；`perfect` 不依赖 DQN。训练和普通 AI 演示默认使用 500 步上限，达到上限单独记为超时；长局对照可提高 `--max-steps`。

<a id="radar-iq"></a>
## radar-iq · 雷达脉内调制识别

### 功能与入口

输入一段雷达脉冲的 IQ 数据，判断其调制类别。Web 工作台支持合成样例、IQ 文件上传、时频图与波形、类别分数、实验评估及结果导出。

| 路径 | 作用 |
| --- | --- |
| `aimc/signals.py` / `data.py` | 合成信号、输入格式校验、数据整理与时频图编码 |
| `aimc/engine.py` | 模型训练、评估、权重加载与推理 |
| `aimc/web.py` / `web/` | Flask 接口与浏览器工作台 |
| `aimc/resources.py` | 训练资源与计算占空比控制 |
| `aimc/upstream.py` / `third_party/` | 固定版本上游模型及谱图实现 |
| `outputs/stft_cnn/` | 8 类合成实验的权重、参数、指标和图表 |
| `outputs/official_demo/` | 30 类官方小样本实验的权重、参数、指标和图表 |

当前源码依赖 Linux 资源接口，以下在 **Linux / WSL** 中执行。先进入 `radar-iq/`，再配置环境和启动：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -B -m aimc.web --port 8080
```

默认访问 [http://127.0.0.1:8080](http://127.0.0.1:8080)。Web 工作台使用保留的权重进行推理，启动时不训练模型，也不下载数据。

### 识别原理

处理流程为：

```text
IQ 信号 → 分帧加 Hann 窗 → STFT 功率谱 → 归一化与缩放 → CNN → 类别分数
```

IQ 的两个分量组成复数信号 `x[n] = I[n] + jQ[n]`。STFT 使用 64 点窗、32 点步长和 64 点 FFT，分析短时间段内的频率组成，再把功率映射到 40 dB 动态范围，缩放为 96 × 96 时频图。横轴表示时间，纵轴表示频率，亮度表示相对功率。

CNN 通过两组卷积、ReLU 与池化学习局部图案，经自适应池化和全连接层输出类别得分。训练使用带标签样本与交叉熵损失，按验证集损失选择权重；推理时使用 Softmax 给出相对分数。该分数未经置信度校准，不能直接当作判断正确的概率。

### 模型与实验范围

| 项目 | 8 类合成模型 | 30 类官方小样本模型 |
| --- | --- | --- |
| 权重 | `outputs/stft_cnn/best.pt` | `outputs/official_demo/best.pt` |
| 单脉冲长度 | 1024 点 | 1000 点 |
| 采样率 | 1 MHz | 50 MHz |
| 附加预处理 | 未启用论文式迭代预处理 | 6 次方差预处理 |
| 测试样本 | 800 条 | 7500 条加噪样本 |
| 测试准确率 | 71.00% | 7.61% |
| Macro-F1 | 0.7058 | 0.0654 |

8 类合成信号包括 `CW`、`LFM_UP`、`LFM_DOWN`、`NLFM`、`BPSK`、`BFSK`、`4FSK` 和 `SFM`，载频、带宽、初相、幅度或码元序列随随机种子变化。

30 类实验从已下载的官方测试脉冲中重新划分训练、验证和测试集合，并在各集合内生成加噪样本，属于小规模流程演示，不是完整官方训练集实验，也不是官方固定测试协议的基准结果。其识别效果较弱。当前目录保留模型和正式评估产物，**未包含原始 IQ 数据与 `data/` 数据集**。

### IQ 输入与新样例

| 格式 | 支持的内容 |
| --- | --- |
| `.npy` | 复数 `(N,)` / `(B,N)`，或实数 `(2,N)`、`(N,2)`、`(B,2,N)`、`(B,N,2)` |
| `.npz` | `iq` 字段采用上述形状；可附带标量 `sample_rate_hz` |
| `.csv` | 无表头两列 `I,Q`，表示一个脉冲 |

Web 上传每次最多 64 条脉冲、8 MB。程序检查长度、采样率、有限值及全零脉冲；不会自动裁切或重采样来适配模型。浏览器识别记录保存预测元数据，不保存上传的原始 IQ。

在已配置好的 Linux / WSL 环境、`radar-iq/` 中生成并识别新样例：

```bash
.venv/bin/python -m aimc sample --class LFM_UP --snr 6 --seed 2026 --output data/new_pulse.npy
.venv/bin/python -m aimc predict --input data/new_pulse.npy --sample-rate 1000000 --checkpoint outputs/stft_cnn/best.pt --device cpu --threads 1
```

这些样例是程序生成的模拟信号，不是新增的实测数据。更换类别、信噪比或随机种子可以生成不同样本。

<a id="web-chat"></a>
## web-chat · 在线聊天室

### 启动与使用

```powershell
Set-Location .\web-chat
python server.py
```

默认访问 [http://127.0.0.1:8000](http://127.0.0.1:8000)。服务实际监听 `0.0.0.0`，同时托管 `static/` 页面和 JSON API；通过环境变量 `PORT` 改变端口。

用户填写昵称后，可创建房间、选择是否启用六位数字密码，也可从房间列表或凭密码加入。支持文本消息、成员列表、房主管理，以及管理员房间与历史数据查看。管理员用户名初始为 `admin`，初始密码可由 `ADMIN_PASSWORD` 环境变量设置；已有持久化账号会从 `admin.json` 加载。

| 文件 | 作用 |
| --- | --- |
| `server.py` | HTTP 路由、房间与会话、权限检查、管理员功能 |
| `static/index.html` | 大厅、聊天页面与管理界面 |
| `static/app.js` | 轮询、消息渲染、会话切换和用户操作 |
| `static/style.css` | 页面样式 |

### 通信与状态管理

前端每 1 秒携带 `since` 游标获取房间状态，服务端只返回游标之后的新增消息。采用普通 HTTP 短轮询，便于适配单端口和反向代理部署。

- `pollBusy` 保证同一时刻只有一个状态请求在途，`pollAgain` 记录后续需要补拉的状态。
- 前端根据消息 ID 去重，避免重复响应导致重复显示。
- 切换房间后，比对响应所属会话，丢弃旧会话的过期结果。
- 会话凭证 `token` 与公开成员标识 `mid` 分开使用；管理操作由后端校验权限。

| 运行参数 | 当前值 |
| --- | --- |
| 单房间在线人数上限 | 10 人 |
| 心跳超时 | 15 秒 |
| 单条消息长度上限 | 500 字符 |
| 单房间保留消息数 | 最近 200 条 |
| 空房间回收等待时间 | 300 秒 |

房间、成员、消息和房间历史保存在内存，进程重启后清空。管理员账号会尽力持久化到项目目录的 `admin.json`，它不是完整聊天记录数据库。

<a id="file-search"></a>
## file-search · SQLite 文件查找器

### 启动与搜索

在仓库根目录打开新终端，进入项目后建立独立索引并启动 Web 服务：

```powershell
Set-Location .\file-search
python -m ffinder update .. --db .\index.sqlite
python -m ffinder serve -p .. --db .\index.sqlite --open
```

默认访问 [http://127.0.0.1:8765](http://127.0.0.1:8765)，服务只监听本机。也可使用 `run.bat` 独立启动；该入口采用查找器的默认索引位置与已有根目录配置。

在 `file-search/` 中执行命令行搜索：

```powershell
python -m ffinder report pdf --db .\index.sqlite
python -m ffinder -e py main --db .\index.sqlite
python -m ffinder -t dir config --db .\index.sqlite
python -m ffinder -r '^test_.*\.py$' --db .\index.sqlite
python -m ffinder status --db .\index.sqlite
```

多个关键词采用 AND 语义，只匹配文件名；`-e` 限定扩展名，`-t` 限定文件或目录，`-p` 限定路径范围，`-r` / `-g` 使用正则或通配符。

### Web 结果操作

| 操作 | 行为 |
| --- | --- |
| 单击结果 | 选中当前条目 |
| 双击文件夹 | 用系统文件管理器打开该文件夹 |
| 双击文件 | 只打开文件所在目录，不执行或打开文件本身 |
| `↑` / `↓`、`Enter` | 选择结果，再按回车打开对应目录 |
| 复制按钮 | 复制完整路径，双击按钮不会触发打开目录 |
| `/`、`Esc` | 聚焦搜索；搜索框聚焦时按 Esc 清空 |
| 重建索引 | 重新扫描配置的目录，更新文件名快照 |

Windows 使用系统资源管理器，macOS / Linux 使用对应文件管理器。打开操作受索引根目录范围限制；结果已经删除或移动时，页面提示重新建立索引。

### 索引与查询原理

`ffinder/scan.py` 使用 `os.scandir()` 枚举目录，利用多进程和动态任务拆分分担扫描工作；`index.py` 将路径存入 SQLite。完成一次建索引后，普通查询直接访问数据库，无需重新遍历文件树。

核心结构是 `files` 表和 `files_name` 联合索引：

| 字段或结构 | 作用 |
| --- | --- |
| `path` | 完整路径，作为主键区分同名文件 |
| `name` | 文件名，使用 `NOCASE` 比较规则 |
| `is_dir` | 标识文件或目录 |
| `WITHOUT ROWID` | 以 `path` 主键组织表数据，省去额外隐藏行号结构 |
| `files_name(name, path)` | 先按文件名、再按路径排序，支持文件名前缀范围定位 |
| `meta(key, value)` | 记录根目录、排除项、构建时间和条目数 |

前缀匹配如 `report%` 可以利用 B-tree 范围定位；任意位置匹配如 `%report%` 通常仍需扫描候选条目。查询还读取 `is_dir`，因此不能声称 `(name, path)` 索引总能避免回表。SQLite `NOCASE` 默认主要处理 ASCII 大小写。

索引是快照，文件新增、改名或删除后需要重建。重建先写临时库，成功后原子替换正式库；“删除索引”只删除数据库，不删除真实文件。该项目不检索文件正文。

主要模块：`scan.py` 扫描，`index.py` 建库与检索，`cli.py` 命令行，`server.py` 本地接口，`ffinder/web/index.html` 搜索界面。

<a id="media-player"></a>
## media-player · 音视频播放器

### 启动与操作

已有可用环境时：

```powershell
Set-Location .\media-player
.\run.bat
```

首次配置新环境时，在 `media-player/` 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe player.py
```

也可把媒体文件拖到 `run.bat` 上，或运行 `player.py "完整文件路径"`。窗口支持打开文件、整窗拖放、播放与暂停、停止、进度拖动及音量调整。缺少拖放扩展时，仍可通过“打开文件”按钮选择文件。

| 文件 | 作用 |
| --- | --- |
| `player.py` | Tkinter 界面、媒体播放、进度控制和状态切换 |
| `wav_visualizer.py` | 音频读取、FFmpeg 频谱分析和 Canvas 柱形绘制 |
| `run.bat` | 使用本目录 `.venv/Scripts/pythonw.exe` 启动 |
| `requirements.txt` | ffpyplayer 与 tkinterdnd2 依赖版本 |

### 视频显示与音频频谱

视频由 ffpyplayer 调用 FFmpeg 解码，取得 RGB 帧，经缩放和 Tk 原生图像通路在 Canvas 上显示。界面按钮、图标、进度条与音量条也由 Canvas 绘制，无需额外图像显示库。

音频播放时显示 **48 根从低频到高频排列的渐变频谱柱**，带峰值短线与轻微倒影：

- 复用当前 ffpyplayer 安装中附带的 FFmpeg 可执行文件，以 `showfreqs` 分析真实频率能量；不使用随机动画，也不另加 Python 分析依赖。
- 后台线程读取分析帧，最多缓存 8 帧，由播放时钟选择对应画面；目标分析帧率为 25 FPS。
- 暂停时固定画面，拖动进度后重新分析目标位置；停止、换文件和关闭窗口时释放对应分析进程。
- 柱高映射和起落速度已调整，使常见音乐强弱、节拍变化更明显。
- 频谱来自源文件，调低输出音量不会改变分析结果；静音片段保持低位。

此前已逐项验证频谱分析的格式为 **WAV、MP3、FLAC、M4A、AAC、OGG、OPUS、WMA**。其他音视频格式能否播放或分析，取决于当前 FFmpeg 的实际解码支持；程序中的扩展名列表不等于所有格式都已测试。

媒体解码库和独立的频谱分析程序属于两条通路：新环境即使安装了 ffpyplayer，也需确认其安装中包含对应的 FFmpeg 可执行文件。找不到分析程序或无法分析时，播放器回到普通音频显示方式。

## 文件维护与恢复

各项目的 README 已合并到根目录本文档，项目内部的 Python 模块名和产品名保持原样。模型权重、训练曲线、混淆矩阵、时频图以及程序使用的前端资源均保留。

`备份/20261021_整理前/` 中保留源码与原始文档压缩包、原始文件清单、目录改名记录和清理记录。旧测试目录、性能测试脚本、临时音频样例、重复预测结果及缓存已从当前项目中清理；需要复现旧测试时，从备份压缩包恢复对应文件。虚拟环境内第三方包附带的 README、许可证和依赖源码不属于项目文档清理范围。

当前可直接使用的权重位置：

- `snake-ai/snake_model.pth`
- `radar-iq/outputs/stft_cnn/best.pt`
- `radar-iq/outputs/official_demo/best.pt`

重新训练、重新评估或新增样例会产生新的数据和结果文件；已保留的实验指标描述的是对应权重与既有实验条件，不自动代表后续训练结果。

## 常见问题

| 问题 | 检查方式 |
| --- | --- |
| 命令提示找不到模块 | 确认当前目录正确，并使用已安装对应依赖的 Python；`snake-ai` 优先通过 `uv run` 启动 |
| 雷达项目在 Windows 提示没有 `resource` | 当前资源控制实现采用 Linux 接口，使用 Linux / WSL 环境运行 |
| 雷达上传被拒绝 | 核对所选模型的采样率、长度、文件大小和 IQ 数组形状 |
| 找不到本地 IQ 数据集 | 当前副本未包含原始数据；可先生成合成样例，官方模型需另行准备匹配的数据 |
| 搜索结果过期或目录打不开 | 检查文件是否移动、根目录配置是否正确，再重建索引 |
| 播放器缺少依赖 | 用 `media-player/.venv/Scripts/python.exe -m pip` 检查或安装依赖，避免装到其他 Python 环境 |
| 音频能播放但没有频谱 | 检查当前安装是否带有可执行的 FFmpeg，并确认文件存在可解码的音频流 |
| 端口被占用 | 贪吃蛇、雷达、查找器通过 `--port` 改端口，聊天室通过 `PORT` 环境变量改端口 |

## 上游来源

雷达项目复用 [AIMC-image-classification](https://github.com/seb-cocks/AIMC-image-classification) 的 STFT-CNN `MainModel` 和 `batched_iq_to_power_spectrogram`，源码保存在 `radar-iq/third_party/AIMC-image-classification/`，固定提交为 `6d50291fd3ad29389dff872b3ce73c06fb6f5c7c`。本项目完成数据适配、输入校验、训练与评估流程、推理接口和中文 Web 工作台集成。

相关论文与数据来源记录于上游 `ref.bib`，包括 [AIMC-Spec 论文](https://arxiv.org/abs/2601.08265) 和 [AIMC-Spec-v2 数据集](https://www.kaggle.com/datasets/sebastiancocks/aimc-spec-v2)。来源仓库当前未附独立 LICENSE 文件，保留作者和来源信息，使用和分发应结合上游许可情况确认。
