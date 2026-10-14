# ffind —— 以「快」为第一目标的文件查找器

一次**并行扫描**建立文件名索引,之后所有查询只碰一个 SQLite 文件,
毫秒级返回。适合反复找文件、记不全名字、只想按关键字/类型/路径过滤的场景。

> 与 `find` 的根本区别:`find` **每次查询**都要从根目录重新遍历整棵目录树;
> `ffind` 把遍历成本摊到**一次性建索引**上,查询时不再碰磁盘目录结构。
> 树越大、查得越勤,优势越明显。

## 它为什么快

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

## 安装 / 运行

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

## 用法

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

## Web 界面

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
- **键盘操作**:`/` 聚焦搜索,`↑`/`↓` 选择,`Enter` 用 `xdg-open` 打开,`⧉` 复制路径,`Esc` 清空。
- **零依赖**:Python 标准库 `http.server` + 单个 `web/index.html`(内联 CSS/JS),
  **只监听 `127.0.0.1`**,不对外网暴露。打开文件受限于索引根范围,越界返回 403。

> WSL/远程开发:VS Code 会自动转发 `127.0.0.1:8765`,直接在本机浏览器打开即可。

## 实测性能

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

## 测试

```bash
uv run --no-project --with pytest python -m pytest tests -q
```

覆盖:扫描器的不 stat/提前返回分批/嵌套根去重,查询的前缀优先/多词 AND/
大小写/正则/通配/扩展名/类型/`-p` 路径限定/`-x` 过滤/`LIKE` 元字符转义,
以及 CLI 冒烟和增量重建。

## 设计边界(诚实说明)

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
- Web 界面**只监听 `127.0.0.1`**,不对局域网/公网暴露;`xdg-open` 打开文件
  受限于索引根目录范围,越界路径返回 403。它是本地工具的可视化外壳,不是多用户服务。

## 对比同类

| 工具                 | 机制       | 查询是否需预建索引 | 语言           |
| ------------------ | -------- | --------- | ------------ |
| `find`             | 每次全量遍历   | 否         | C            |
| `fd`               | 每次并行遍历   | 否         | Rust         |
| `locate`/`plocate` | 预建索引,查索引 | 是         | C            |
| **`ffind`**        | 预建索引,查索引 | 是         | Python(纯标准库) |

`ffind` 的定位:`plocate` 的「索引换速度」思路,做成一个**零依赖、可嵌入、
扫描引擎自身也快**的 Python 库 + CLI + 本地 Web 界面。

## 目录结构

```
file-finder/
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
