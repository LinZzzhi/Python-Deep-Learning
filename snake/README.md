# 基于强化学习的贪吃蛇

这是一个面向教学的单文件 Double DQN 贪吃蛇项目，同时提供手动游戏、已训练模型演示和哈密顿环必胜策略。默认 AI 将模型与安全路径规划结合，优先走安全的最短食物路线。核心实现位于 `enhancedSnake.py`；配套 PDF 描述原始 DQN 设计，当前算法改进以代码和本 README 为准。

## 快速开始

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

## 项目结构

- `enhancedSnake.py`：环境、DQN、训练、推理、手动游戏和完美策略。
- `demo_rl_snake.py`：直接播放已有模型的轻量入口。
- `snake_model.pth`：与当前网络结构兼容的已训练权重。
- `training_progress.png`：最近一次训练的得分和损失曲线。
- `基于强化学习的贪吃蛇游戏_详细设计文档.pdf`：教学设计文档。
- `tests/`：环境边界、模型、训练目标、路径规划和 Web 游戏回归测试。

## 测试

测试不打开游戏窗口：

```bash
uv run python -m unittest discover -s tests -v
```

训练环境默认在 500 步后终止，避免经验回放被死循环占据；手动模式和完美策略不使用这个训练上限。AI 演示默认启用哈密顿安全层与 BFS 路径规划：先过滤可能追上尾巴或沿环越过食物的动作，再在保持蛇身环形顺序的空闲区域中搜索最短路线，DQN 在同样短的候选路线中决策。当食物暂时被身体挡住时，选择沿环推进最多的安全动作，等待蛇尾释放空间。每一步重新规划，不使用容易过期的路径缓存。CLI 和 Web 共用这套逻辑。

`--unsafe` 关闭全局安全约束和规划，保留模型与单步合法动作掩码。训练与推理都会排除反向无效动作；新训练使用 Double DQN（在线网络选动作、目标网络估值）和 Huber 损失，以减少价值高估和异常奖励导致的梯度波动。网络仍为 10→64→64→4，仓库已有权重可直接加载。模型权重和历史训练曲线没有被替换；新训练算法的效果需重新训练后评测。

完美策略允许通过 `--max-steps` 人工限制演示时长，但设置上限后不再承诺本局一定完成。

## 优化对比

使用仓库已有模型、默认初始蛇身、食物种子 0–19，每局 500 步：

| 策略 | 平均得分 | 碰撞局数 |
|---|---:|---:|
| 原模型 + 原哈密顿安全层（`shield`） | 17.80 | 0 / 20 |
| 原模型 + 安全最短路径规划（`smart`，默认） | 22.25 | 0 / 20 |

默认策略平均得分提高 25%。使用相同初始蛇身和种子 0–4 的完美策略长局对比，优化前后均为 5 / 5 胜利，铺满棋盘平均步数从 47,789.6 降至 37,597.6，减少约 21.3%。这些是固定种子的样本结果，不代表所有种子的提升幅度。

`evaluate` 支持 `raw`（仅模型与单步掩码）、`shield`（原安全层）、`smart`（新规划器）、`perfect`（无模型规划），默认 `all`。报告包含每局种子、得分、步数、终止原因，以及平均得分、每颗食物步数、胜利次数和碰撞次数；步数上限导致的结束单独计为 `timeout`。
