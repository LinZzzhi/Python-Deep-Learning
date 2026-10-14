"""
基于强化学习的贪吃蛇游戏 —— 整合版（教学用，单文件）

对应《基于强化学习的贪吃蛇游戏 —— 软件详细设计文档》：
    第 3 章  配置设计（config 全局常量区）
    第 4 章  游戏环境设计（SnakeEnv：reset / step / render）
    第 5 章  强化学习核心（状态 10 维、奖励表、DQN、ReplayMemory、ε-贪婪）
    第 6 章  训练流程（train_agent：双网络 + 经验回放 + 曲线绘制）
    第 7 章  演示与运行（play_with_model 模型演示 / play_manual 传统键盘模式）
    扩展      预判规划器（LookaheadPlanner：虚拟蛇预演 + 动态哈密顿锁定）

用法：
    python enhancedSnake.py train [episodes]   # 训练并保存模型（默认 2000 局）
    python enhancedSnake.py play               # 加载模型观看 AI 自动游戏（预判规划）
    python enhancedSnake.py manual             # 传统键盘手动模式（方向键控制）
    python enhancedSnake.py perfect [fps]      # 哈密顿环完美策略，保证铺满棋盘胜利
    python enhancedSnake.py evaluate           # 无窗口对比各策略
"""

import argparse
import json
import os
import random
from collections import Counter, deque
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================
# 3. 配置设计（config）—— 集中管理"魔法数字"
# ============================================================
SCREEN_WIDTH, SCREEN_HEIGHT = 400, 400
GRID_SIZE = 20                        # 每个网格像素大小
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE
FPS = 30

# 方向常量（用元组表示位移，便于向量运算）
UP, DOWN, LEFT, RIGHT = (0, -1), (0, 1), (-1, 0), (1, 0)
DIRECTIONS = (UP, DOWN, LEFT, RIGHT)  # 索引即动作编号：0=上 1=下 2=左 3=右

# 状态 / 动作维度（5.2 状态空间：10 维向量；5.3 动作：4 个方向）
STATE_SIZE = 10
ACTION_SIZE = 4

# 奖励设计（5.3 动作与奖励）
REWARD_FOOD = 20.0       # 吃到食物：强烈正向
REWARD_DEATH = -20.0     # 撞墙/撞身：强烈惩罚
REWARD_CLOSER = 1.0      # 靠近食物：引导方向
REWARD_FARTHER = -1.0    # 远离食物：防止绕路
REWARD_IDLE = -0.1       # 原地不动：鼓励行动
REWARD_TIMEOUT = -10.0   # 超时（防死循环）：限制单局时长
REWARD_WIN = 100.0       # 胜利：蛇身覆盖全部格子且首尾相连
TOTAL_CELLS = GRID_WIDTH * GRID_HEIGHT   # 400 格，铺满即胜利

MAX_STEPS_PER_EPISODE = 500           # 单局最大步数（超时判负）

# 网格邻接表：路径规划会反复查询邻居，预先算好免去重复的边界判断
ALL_CELLS = tuple((x, y) for x in range(GRID_WIDTH) for y in range(GRID_HEIGHT))
_NEIGHBORS = {
    (x, y): tuple((x + dx, y + dy) for dx, dy in DIRECTIONS
                  if 0 <= x + dx < GRID_WIDTH and 0 <= y + dy < GRID_HEIGHT)
    for x, y in ALL_CELLS
}

# 预判规划器（LookaheadPlanner）
LOOKAHEAD_CONNECT_LENGTH = 100   # 蛇长达到后，吃完食物还要求空白区域保持连通
LOOKAHEAD_LOCK_LENGTH = 200      # 蛇长达到后，尝试把蛇身锁进动态哈密顿环
LOOKAHEAD_LOCK_BUDGET = 3000     # 单次哈密顿路径搜索的节点上限
LOOKAHEAD_LOCK_RETRY = 10        # 搜索耗尽预算后，间隔多少步再试
LOOKAHEAD_STALL_STEPS = TOTAL_CELLS   # 连续这么多步没吃到食物，视为陷入周旋

# DQN 超参数
LEARNING_RATE = 0.001
GAMMA = 0.99                          # 折扣因子
BATCH_SIZE = 64
MEMORY_CAPACITY = 10000
EPSILON_START, EPSILON_END, EPSILON_DECAY = 1.0, 0.01, 0.995
TARGET_UPDATE = 100                   # 目标网络更新频率

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "snake_model.pth"
PROGRESS_PATH = PROJECT_DIR / "training_progress.png"


# ============================================================
# 4. 游戏环境设计（SnakeEnv）—— 核心类
# ============================================================
class SnakeEnv:
    """负责"游戏世界"的全部规则：蛇、食物、移动、碰撞、绘制。

    遵循强化学习环境接口（reset / step），可被智能体调用。
    """

    def __init__(self, render=False, max_steps=MAX_STEPS_PER_EPISODE, rng=None):
        if max_steps is not None and max_steps <= 0:
            raise ValueError("max_steps 必须是正整数或 None")
        self.render_enabled = render
        self.max_steps = max_steps
        self.rng = rng if rng is not None else random
        self.screen = None
        if render:
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("贪吃蛇 - 强化学习")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 22)
        self.reset()

    # ---------- 4.2 核心属性 ----------
    def reset(self):
        """重置蛇身（初始长度 3）、方向、分数、食物，返回初始状态。"""
        cx, cy = GRID_WIDTH // 2, GRID_HEIGHT // 2
        self.snake = [(cx, cy), (cx - 1, cy), (cx - 2, cy)]  # 索引 0 为蛇头
        self.direction = RIGHT
        self.score = 0
        self.steps = 0
        self.alive = True
        self.won = False
        self.timed_out = False
        self.termination_reason = None
        self._spawn_food()
        return self._get_state()

    def _spawn_food(self):
        """在所有空闲格子中随机放置食物。"""
        occupied = set(self.snake)
        free = [(x, y) for x in range(GRID_WIDTH)
                         for y in range(GRID_HEIGHT)
                         if (x, y) not in occupied]
        self.food = self.rng.choice(free) if free else None

    def _is_danger(self, direction):
        """蛇头沿 direction 走一步是否危险（撞墙或撞身），0/1。"""
        hx, hy = self.snake[0]
        nx, ny = hx + direction[0], hy + direction[1]
        if nx < 0 or nx >= GRID_WIDTH or ny < 0 or ny >= GRID_HEIGHT:
            return 1
        # 未吃食物时尾格会同步让出，因此可以安全进入。
        body = self.snake if (nx, ny) == self.food else self.snake[:-1]
        return 1 if (nx, ny) in body else 0

    def _get_state(self):
        """5.2 状态空间：10 维向量。

        维 1-2 ：食物相对蛇头的方向（归一化 dx, dy）
        维 3-6 ：上/下/左/右 是否危险（墙或自身，0/1）
        维 7-10：当前移动方向（one-hot：上/下/左/右）
        """
        hx, hy = self.snake[0]
        if self.food is None:
            food_dx = food_dy = 0.0
        else:
            fx, fy = self.food
            food_dx = (fx - hx) / GRID_WIDTH
            food_dy = (fy - hy) / GRID_HEIGHT
        danger = [self._is_danger(d) for d in DIRECTIONS]
        one_hot = [1 if self.direction == d else 0 for d in DIRECTIONS]
        return np.array([food_dx, food_dy] + danger + one_hot, dtype=np.float32)

    def step(self, action):
        """强化学习核心接口：执行一步并返回 (next_state, reward, done)。"""
        if not isinstance(action, (int, np.integer)) or not 0 <= action < ACTION_SIZE:
            raise ValueError(f"action 必须是 0 到 {ACTION_SIZE - 1} 的整数")
        if not self.alive or self.won:
            return self._get_state(), 0.0, True

        self.steps += 1
        new_dir = DIRECTIONS[action]
        reward = 0.0

        # 反向 = 无效动作：原地不动，小惩罚鼓励行动
        if (new_dir[0] + self.direction[0], new_dir[1] + self.direction[1]) == (0, 0):
            reward += REWARD_IDLE
            done = self._has_timed_out()
            if done:
                reward += REWARD_TIMEOUT
            return self._get_state(), reward, done

        self.direction = new_dir
        hx, hy = self.snake[0]
        new_head = (hx + new_dir[0], hy + new_dir[1])

        # 4.4 碰撞检测：本质是"坐标判定"
        old_dist = abs(hx - self.food[0]) + abs(hy - self.food[1])
        new_dist = abs(new_head[0] - self.food[0]) + abs(new_head[1] - self.food[1])

        done = False
        hit_wall = not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT)
        # 吃食物时尾巴不收缩，需把尾巴也算进碰撞范围
        hit_self = new_head in (self.snake if new_head == self.food else self.snake[:-1])
        if hit_wall or hit_self:
            done = True
            self.alive = False
            self.termination_reason = "wall" if hit_wall else "self"
            reward += REWARD_DEATH
        else:
            self.snake.insert(0, new_head)
            if new_head == self.food:
                self.score += 1
                reward += REWARD_FOOD
                if len(self.snake) == TOTAL_CELLS:
                    # 胜利条件：蛇身覆盖 400×400 全部 400 格，且首尾相连
                    done = True
                    head = self.snake[0]
                    tail = self.snake[-1]
                    self.won = abs(head[0] - tail[0]) + abs(head[1] - tail[1]) == 1
                    if self.won:
                        self.termination_reason = "win"
                        reward += REWARD_WIN
                    else:
                        self.alive = False
                        self.termination_reason = "full_board"
                    self.food = None      # 盘面已满，不再放置食物
                else:
                    self._spawn_food()
            else:
                self.snake.pop()
            # 奖励塑形：靠近 +1 / 远离 -1，引导蛇主动接近食物
            if new_dist < old_dist:
                reward += REWARD_CLOSER
            elif new_dist > old_dist:
                reward += REWARD_FARTHER
            if not done and self._has_timed_out():
                done = True
                reward += REWARD_TIMEOUT

        return self._get_state(), reward, done

    def _has_timed_out(self):
        """达到当前环境的步数上限时，将终止状态同步到环境。"""
        if self.max_steps is None or self.steps < self.max_steps:
            return False
        self.timed_out = True
        self.alive = False
        self.termination_reason = "timeout"
        return True

    def render(self, plan=None):
        """用 Pygame 绘制网格、蛇、食物、分数、步数；plan 为 AI 规划路线。"""
        if not self.render_enabled:
            return
        self.screen.fill((240, 240, 245))
        for x in range(0, SCREEN_WIDTH, GRID_SIZE):
            pygame.draw.line(self.screen, (226, 226, 232), (x, 0), (x, SCREEN_HEIGHT))
        for y in range(0, SCREEN_HEIGHT, GRID_SIZE):
            pygame.draw.line(self.screen, (226, 226, 232), (0, y), (SCREEN_WIDTH, y))

        for x, y in plan or ():
            pygame.draw.circle(self.screen, (120, 160, 230),
                               (x * GRID_SIZE + GRID_SIZE // 2,
                                y * GRID_SIZE + GRID_SIZE // 2), 3)

        if self.food is not None:
            fx, fy = self.food
            pygame.draw.rect(self.screen, (220, 60, 60),
                             (fx * GRID_SIZE + 3, fy * GRID_SIZE + 3,
                              GRID_SIZE - 6, GRID_SIZE - 6), border_radius=6)
        for i, (x, y) in enumerate(self.snake):
            color = (60, 180, 90) if i == 0 else (40, 140, 70)
            pygame.draw.rect(self.screen, color,
                             (x * GRID_SIZE + 1, y * GRID_SIZE + 1,\
                              GRID_SIZE - 2, GRID_SIZE - 2), border_radius=5)

        text = self.font.render(f"Score: {self.score}  Steps: {self.steps}",
                                True, (40, 40, 60))
        self.screen.blit(text, (6, 4))
        if self.won:
            win = self.font.render("YOU WIN! 400/400 head-tail linked",
                                   True, (40, 140, 70))
            self.screen.blit(win, (SCREEN_WIDTH // 2 - 160, SCREEN_HEIGHT // 2))
        elif self.timed_out:
            over = self.font.render("Time limit reached!  SPACE to restart",
                                    True, (180, 110, 30))
            self.screen.blit(over, (SCREEN_WIDTH // 2 - 150, SCREEN_HEIGHT // 2))
        elif not self.alive:
            over = self.font.render("Game Over!  SPACE to restart",
                                    True, (180, 40, 40))
            self.screen.blit(over, (SCREEN_WIDTH // 2 - 130, SCREEN_HEIGHT // 2))
        pygame.display.flip()


class DirectionBuffer:
    """手动输入缓冲：每个时间步只消费一个方向，新方向与"上一个排队方向"比较。

    直接修改 env.direction 时，同一帧内快速连按两键（向右走时按 ↑ 再按 ←）
    会绕过反向检查，让蛇头掉头撞进脖子；排队后两次转向分两步执行。
    """

    def __init__(self, capacity=2):
        self.capacity = capacity
        self.pending = deque()

    def push(self, env, direction):
        last = self.pending[-1] if self.pending else env.direction
        if len(self.pending) >= self.capacity or direction == last or \
                (direction[0] + last[0], direction[1] + last[1]) == (0, 0):
            return False
        self.pending.append(direction)
        return True

    def peek_action(self, env):
        return DIRECTIONS.index(self.pending[0] if self.pending else env.direction)

    def next_action(self, env):
        action = self.peek_action(env)
        if self.pending:
            self.pending.popleft()
        return action


# ============================================================
# 5.4 DQN 网络结构：三层全连接拟合 Q 值函数
# ============================================================
class DQN(nn.Module):
    """输入状态（10 维），输出 4 个动作的 Q 值。"""

    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)   # 输出 4 个动作 Q 值

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# ============================================================
# 5.5 经验回放（ReplayMemory）
# ============================================================
class ReplayMemory:
    """随机存储历史经验 (s, a, r, s', done)，训练时随机采样一批，
    打破样本相关性、提升稳定性。deque 容量 10000，满后自动淘汰旧经验。"""

    def __init__(self, capacity=MEMORY_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.tensor(np.array(states), dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(np.array(next_states), dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))

    def __len__(self):
        return len(self.buffer)


# ============================================================
# 5.6 ε-贪婪策略 + 智能体（DQNAgent）
# ============================================================
class DQNAgent:
    """决策 + 学习：双网络（policy_net 训练 / target_net 稳定目标）。"""

    def __init__(self):
        self.policy_net = DQN(STATE_SIZE, ACTION_SIZE)
        self.target_net = DQN(STATE_SIZE, ACTION_SIZE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory()
        self.epsilon = EPSILON_START
        self.learn_steps = 0

    def select_action(self, state, training=True, allowed_actions=None):
        """探索：以概率 ε 随机选动作；利用：选当前 Q 值最大的动作。

        动作掩码：状态维 3-6 本身就是"四向是否危险"，直接排除
        "下一步必死"的动作，显著减少撞墙/撞身，加速收敛。
        """
        if allowed_actions is None:
            allowed = reflex_actions(state)
        else:
            allowed = list(dict.fromkeys(allowed_actions))
            if not allowed or any(a not in range(ACTION_SIZE) for a in allowed):
                raise ValueError("allowed_actions 必须包含有效动作")
        if training and random.random() < self.epsilon:
            return random.choice(allowed)
        if len(allowed) == 1:
            return allowed[0]
        q = self.q_values(state)
        return int(max(allowed, key=lambda a: q[a]))

    def q_values(self, state):
        """返回 4 个动作的 Q 值列表，供决策与"AI 思维"可视化共用。"""
        with torch.no_grad():
            tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            return self.policy_net(tensor)[0].tolist()

    def learn(self):
        """经验回放采样 → Double DQN 目标 → Huber 损失 → Adam 更新。

        策略网络选择合法动作，目标网络评估该动作，减少 Q 值高估。
        池中样本不足 batch 时不训练，返回 None。
        """
        if len(self.memory) < BATCH_SIZE:
            return None
        states, actions, rewards, next_states, dones = self.memory.sample(BATCH_SIZE)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = _masked_double_q(self.policy_net(next_states),
                                      self.target_net(next_states), next_states)
            target_q = rewards + GAMMA * next_q * (1.0 - dones)

        loss = nn.functional.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # 双网络定期同步，防止 Q 值震荡
        self.learn_steps += 1
        if self.learn_steps % TARGET_UPDATE == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return float(loss.item())

    def update_epsilon(self):
        """训练初期 ε 大（多探索），随训练衰减至 ε_end（多利用）。"""
        self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

    def save(self, path=MODEL_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        torch.save(self.policy_net.state_dict(), temporary_path)
        temporary_path.replace(path)

    def load(self, path=MODEL_PATH):
        self.policy_net.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.epsilon = 0.0            # 演示时纯利用，不再探索，避免"随机送死"


def reflex_actions(state):
    """单步动作掩码：排除反向与下一步必死的方向；无路可走时只排除反向。"""
    dangers = state[2:6]
    reverse = (1, 0, 3, 2)[int(np.argmax(state[6:10]))]
    return [a for a in range(ACTION_SIZE) if dangers[a] == 0 and a != reverse] \
        or [a for a in range(ACTION_SIZE) if a != reverse]


def _mask_q_values(q_values, states):
    """统一训练与推理的动作约束，排除碰撞和无效反向动作。"""
    dangers = states[:, 2:6].bool()
    reverse = states[:, 6:10][:, [1, 0, 3, 2]].bool()
    dangers = dangers | reverse
    # 围困状态仍排除反向动作，并避免一整行 -inf 污染训练目标。
    effective_mask = (dangers & ~dangers.all(dim=1, keepdim=True)) | reverse
    return q_values.masked_fill(effective_mask, -torch.inf)


def _masked_max_q(q_values, states):
    """计算合法动作的最大 Q 值（保留用于 DQN 对照实验）。"""
    return _mask_q_values(q_values, states).max(dim=1).values


def _masked_double_q(policy_values, target_values, states):
    """用在线网络选动作、目标网络估值，避免直接最大化目标网络。"""
    actions = _mask_q_values(policy_values, states).argmax(dim=1, keepdim=True)
    return target_values.gather(1, actions).squeeze(1)


# ============================================================
# 6. 训练流程设计（train_agent）
# ============================================================
def train_agent(episodes=2000, render=False, model_path=MODEL_PATH,
                progress_path=PROGRESS_PATH, seed=42, progress_cb=None):
    """将环境、智能体、经验回放串联，完成完整训练闭环。

    progress_cb(episode, score, best, epsilon, loss) 每局调用一次，供
    Web 控制台等外部监控实时展示进度；回调返回 False 则提前停止训练。
    """
    if episodes <= 0:
        raise ValueError("episodes 必须是正整数")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = SnakeEnv(render=render, max_steps=MAX_STEPS_PER_EPISODE)
    agent = DQNAgent()
    score_history, loss_history = [], []
    best_score = -1

    for episode in range(1, episodes + 1):
        state = env.reset()
        episode_loss, loss_count = 0.0, 0

        while True:
            if render:                       # 可视化训练时处理窗口事件并限速
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                env.render()
                env.clock.tick(FPS)

            action = agent.select_action(state, training=True)
            next_state, reward, done = env.step(action)
            agent.memory.push(state, action, reward, next_state, done)

            loss = agent.learn()             # 存入经验池 + replay 更新
            if loss is not None:
                episode_loss += loss
                loss_count += 1
            state = next_state
            if done:                         # done? 或超时 → 本局结束
                break

        # 记录 score/loss，更新 epsilon
        agent.update_epsilon()
        score_history.append(env.score)
        loss_history.append(episode_loss / loss_count if loss_count else 0.0)

        if env.score > best_score:           # score > best? 保存最佳模型
            best_score = env.score
            agent.save(model_path)

        if progress_cb is not None and progress_cb(
                episode, env.score, best_score,
                agent.epsilon, loss_history[-1]) is False:
            print(f"训练在第 {episode} 局被外部请求提前停止")
            break

        if episode % 10 == 0:                # 每 10 局打印进度
            recent = score_history[-10:]
            print(f"Episode {episode:5d} | score {env.score:3d} | "
                  f"avg10 {sum(recent) / len(recent):5.1f} | "
                  f"best {best_score:3d} | eps {agent.epsilon:.3f} | "
                  f"loss {loss_history[-1]:.3f}")

    print(f"训练结束，最佳得分 {best_score}，模型已保存至 {model_path}")
    _plot_progress(score_history, loss_history, progress_path)
    if render:
        pygame.quit()
    return score_history, loss_history


def _plot_progress(score_history, loss_history, path=PROGRESS_PATH):
    """训练结束后自动绘制 得分曲线 与 损失曲线（training_progress.png）。"""
    import matplotlib
    matplotlib.use("Agg")                    # 无界面环境也能保存图表
    import matplotlib.pyplot as plt

    def moving_avg(values, window=50):
        values = np.array(values, dtype=float)
        kernel = np.ones(window) / window
        return np.convolve(values, kernel, mode="valid")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(score_history, alpha=0.3, label="raw")
    if len(score_history) >= 50:
        axes[0].plot(range(49, len(score_history)),
                     moving_avg(score_history), label="moving avg(50)")
    axes[0].set_title("Score per Episode")
    axes[0].set_xlabel("episode")
    axes[0].set_ylabel("score")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(loss_history, alpha=0.3, label="raw")
    if len(loss_history) >= 50:
        axes[1].plot(range(49, len(loss_history)),
                     moving_avg(loss_history), label="moving avg(50)")
    axes[1].set_title("Loss per Episode")
    axes[1].set_xlabel("episode")
    axes[1].set_ylabel("loss")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"训练曲线已保存至 {path}")


# ============================================================
# 7. 演示与运行
# ============================================================
def play_with_model(model_path=MODEL_PATH, games=5, max_steps=500,
                    planner="lookahead"):
    """7.2 仅演示：加载已训练模型观看 AI 自动游戏。

    planner："lookahead" 预判规划（默认）、"smart" 哈密顿捷径、None 仅 DQN。
    """
    if games <= 0 or max_steps <= 0:
        raise ValueError("games 和 max_steps 必须是正整数")
    if not Path(model_path).is_file():
        print(f"未找到模型文件 {model_path}，请先运行：python enhancedSnake.py train")
        return

    env = SnakeEnv(render=True, max_steps=max_steps)
    try:
        agent = DQNAgent()
        agent.load(model_path)

        for game in range(1, games + 1):
            state = env.reset()
            lookahead = LookaheadPlanner()
            if planner == "smart":
                cycle_idx, _ = _cycle_aligned_with_snake(env)
            plan = None
            while env.alive and not env.won:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return
                env.render(plan)
                env.clock.tick(FPS)
                allowed = None
                if planner == "lookahead":
                    allowed = lookahead.safe_actions(env)
                elif planner == "smart":
                    allowed = _smart_safe_actions(env, cycle_idx)
                if allowed == []:
                    env.alive = False
                    env.termination_reason = "trapped"
                    env.render()
                    break
                action = agent.select_action(
                    state, training=False, allowed_actions=allowed)
                if planner == "lookahead":
                    plan = lookahead.routes.get(action)
                state, _, done = env.step(action)
                if done:
                    env.render()
                    break
            labels = {
                "timeout": "达到步数上限",
                "win": "胜利",
                "wall": "撞墙",
                "self": "撞到自身",
                "trapped": "无安全路径",
            }
            result = labels.get(env.termination_reason, "结束")
            print(f"第 {game} 局：得分 {env.score}，步数 {env.steps}，{result}")
    finally:
        pygame.quit()


def play_manual():
    """7.3 传统手动模式：用键盘事件决定 action，还原经典玩法。"""
    key_to_dir = {pygame.K_UP: UP, pygame.K_DOWN: DOWN,
                  pygame.K_LEFT: LEFT, pygame.K_RIGHT: RIGHT}
    env = SnakeEnv(render=True, max_steps=None)
    env.reset()
    keys = DirectionBuffer()
    print("方向键控制移动，空格重新开始，关闭窗口退出。")

    try:
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        env.reset()
                        keys = DirectionBuffer()
                    elif event.key in key_to_dir and env.alive:
                        keys.push(env, key_to_dir[event.key])  # 缓冲内禁止 180° 掉头
            if env.alive:
                env.step(keys.next_action(env))
            env.render()
            env.clock.tick(FPS)
    finally:
        pygame.quit()


# ============================================================
# 完美策略：哈密顿环 + 安全捷径（数学保证胜利）
# ============================================================
def _build_hamiltonian_cycle():
    """在 20×20 网格上构造哈密顿环，返回 (idx, order)。

    idx[cell] = 格子在环上的索引；order[i] = 环上第 i 个格子。
    路线：第 0 列自上而下 → 第 19..1 行蛇形往返
    → 第 0 行自右向左回原点。
    """
    order = [(0, y) for y in range(GRID_HEIGHT)]
    for row in range(GRID_HEIGHT - 1, 0, -1):
        left_to_right = (GRID_HEIGHT - 1 - row) % 2 == 0
        cols = range(1, GRID_WIDTH) if left_to_right \
            else range(GRID_WIDTH - 1, 0, -1)
        order.extend((x, row) for x in cols)
    order.extend((x, 0) for x in range(GRID_WIDTH - 1, 0, -1))
    return {cell: i for i, cell in enumerate(order)}, order


def _cycle_aligned_with_snake(env):
    """返回与当前蛇头前进方向一致的哈密顿环。"""
    _, order = _build_hamiltonian_cycle()
    for candidate in (order, list(reversed(order))):
        idx = {cell: i for i, cell in enumerate(candidate)}
        head_index = idx[env.snake[0]]
        if env.snake[1] == candidate[(head_index - 1) % TOTAL_CELLS]:
            return idx, candidate
    raise ValueError("当前蛇身未与哈密顿环对齐")


def _prepare_hamiltonian_snake(env, idx, order):
    """让初始蛇身与环的前进方向一致，保持安全性不变量。"""
    head_index = idx[env.snake[0]]
    env.snake = [order[head_index], order[(head_index - 1) % TOTAL_CELLS],
                 order[(head_index - 2) % TOTAL_CELLS]]
    head, neck = env.snake[:2]
    env.direction = (head[0] - neck[0], head[1] - neck[1])
    env._spawn_food()


def _hamiltonian_safe_actions(env, idx):
    """返回不越过食物、不追上尾巴的动作集合。"""
    if env.food is None:
        return []
    head, tail = env.snake[0], env.snake[-1]
    distance_to_tail = (idx[tail] - idx[head]) % TOTAL_CELLS
    distance_to_food = (idx[env.food] - idx[head]) % TOTAL_CELLS
    actions = []

    for action, direction in enumerate(DIRECTIONS):
        if (direction[0] + env.direction[0],
                direction[1] + env.direction[1]) == (0, 0):
            continue
        cell = (head[0] + direction[0], head[1] + direction[1])
        if not (0 <= cell[0] < GRID_WIDTH and 0 <= cell[1] < GRID_HEIGHT):
            continue

        distance = (idx[cell] - idx[head]) % TOTAL_CELLS
        if distance == 0 or distance > distance_to_food:
            continue
        eating = cell == env.food
        tail_is_safe = distance < distance_to_tail or (
            distance == distance_to_tail and not eating)
        if tail_is_safe:
            actions.append(action)
    return actions


def _smart_safe_actions(env, idx):
    """优先选择安全最短食物路径；不可达时尽量沿环向前推进。

    BFS 只在头到尾的空闲环段中搜索，并要求每条边沿环前进。
    路径不会越过食物或破坏蛇身顺序，吃到食物后仍可安全跟随环。
    搜索保守地阻挡当前身体，仅允许已经移走的尾格。
    """
    allowed = _hamiltonian_safe_actions(env, idx)
    if len(allowed) <= 1:
        return allowed

    head = env.snake[0]
    head_index = idx[head]
    food_distance = (idx[env.food] - head_index) % TOTAL_CELLS
    tail_distance = (idx[env.snake[-1]] - head_index) % TOTAL_CELLS
    next_cells = {
        action: (head[0] + DIRECTIONS[action][0],
                 head[1] + DIRECTIONS[action][1])
        for action in allowed
    }

    if food_distance < tail_distance:
        distances = _cycle_food_distances(env, idx)
        reachable = [action for action in allowed if next_cells[action] in distances]
        if reachable:
            shortest = min(distances[next_cells[action]] for action in reachable)
            return [action for action in reachable
                    if distances[next_cells[action]] == shortest]

    # 食物位于身体占用的环段之后时，推进蛇尾释放空间，再重新规划。
    return [max(allowed, key=lambda action:
                (idx[next_cells[action]] - head_index) % TOTAL_CELLS)]


def _cycle_food_distances(env, idx):
    """从食物反向 BFS，一次搜索即可比较所有候选首步。

    只经过头尾之间的空闲环段，且每条边都沿环前进。
    """
    head_index = idx[env.snake[0]]
    blocked = set(env.snake[:-1])
    distances = {env.food: 0}
    queue = deque([env.food])
    while queue:
        cell = queue.popleft()
        forward = (idx[cell] - head_index) % TOTAL_CELLS
        for dx, dy in DIRECTIONS:
            previous = (cell[0] + dx, cell[1] + dy)
            if previous not in idx or previous in blocked or previous in distances:
                continue
            previous_forward = (idx[previous] - head_index) % TOTAL_CELLS
            if 0 < previous_forward < forward:
                distances[previous] = distances[cell] + 1
                queue.append(previous)
    return distances


def _cycle_routes(env, idx, actions):
    """还原哈密顿捷径规划的完整路线，供可视化；食物暂不可达时只含下一格。"""
    head = env.snake[0]
    head_index = idx[head]
    distances = _cycle_food_distances(env, idx) if env.food is not None else {}
    routes = {}
    for action in actions:
        cell = (head[0] + DIRECTIONS[action][0], head[1] + DIRECTIONS[action][1])
        route = [cell]
        while cell in distances and cell != env.food:
            forward = (idx[cell] - head_index) % TOTAL_CELLS
            cell = next(nb for nb in _NEIGHBORS[cell]
                        if distances.get(nb) == distances[cell] - 1
                        and (idx[nb] - head_index) % TOTAL_CELLS > forward)
            route.append(cell)
        routes[action] = route
    return routes


def _select_hamiltonian_action(env, idx, order):
    """在安全最短路径中选择沿环推进最快的动作。"""
    allowed = _smart_safe_actions(env, idx)
    if not allowed:
        raise ValueError("当前蛇身没有安全的哈密顿动作")
    head = env.snake[0]
    return max(allowed, key=lambda action:
               (idx[(head[0] + DIRECTIONS[action][0],
                     head[1] + DIRECTIONS[action][1])] - idx[head]) % TOTAL_CELLS)


def play_perfect(fps=120, max_steps=None):
    """胜利演示：沿哈密顿环走并尽可能抄安全捷径。

    候选格必须位于头到尾之间的空闲环段，而且不能沿环越过食物；不吃时
    可以踩即将让出的尾格。由此保证蛇身不自撞、食物不被反复跳过，最终
    铺满 400 格且首尾相连。
    """
    if fps <= 0 or (max_steps is not None and max_steps <= 0):
        raise ValueError("fps 和 max_steps 必须是正整数")
    env = SnakeEnv(render=True, max_steps=None)
    idx, order = _build_hamiltonian_cycle()
    _prepare_hamiltonian_snake(env, idx, order)

    steps = 0
    running = True
    try:
        while running and (max_steps is None or steps < max_steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            if not running:
                break

            action = _select_hamiltonian_action(env, idx, order)
            _, _, done = env.step(action)
            env.render()
            env.clock.tick(fps)
            steps += 1
            if done:
                break

        result = "胜利：铺满 400 格且首尾相连！" if env.won else "本局结束"
        print(f"{result} 得分 {env.score}，步数 {steps}")
    finally:
        pygame.quit()


# ============================================================
# 预判规划器：虚拟蛇预演 + 追尾周旋 + 动态哈密顿锁定
# ============================================================
def _legal_moves(env):
    """不反向、不出界、不撞身体的动作 → 下一格；未吃食物时尾格会让出。"""
    head = env.snake[0]
    body = set(env.snake[:-1])
    moves = {}
    for action, (dx, dy) in enumerate(DIRECTIONS):
        if (dx + env.direction[0], dy + env.direction[1]) == (0, 0):
            continue
        cell = (head[0] + dx, head[1] + dy)
        if cell in _NEIGHBORS and cell not in body:
            moves[action] = cell
    return moves


def _advance(snake, route, food):
    """虚拟蛇沿 route 前进后的身体；只有终点可能是食物，吃到时增长一节。"""
    grow = 1 if route[-1] == food else 0
    return (route[::-1] + snake)[:len(snake) + grow]


def _survey(body):
    """从蛇头 BFS（除尾格外的身体都是障碍），返回父节点表。

    尾格在表中 ⇔ 蛇头能回到蛇尾；表的大小用于判断空白区域是否连通。
    """
    head = body[0]
    blocked = set(body[:-1])
    parent = {head: None}
    queue = deque([head])
    while queue:
        cell = queue.popleft()
        for nb in _NEIGHBORS[cell]:
            if nb not in parent and nb not in blocked:
                parent[nb] = cell
                queue.append(nb)
    return parent


def _trace(parent, target):
    """沿父节点表还原从起点（不含）到 target 的路线。"""
    route = []
    while parent[target] is not None:
        route.append(target)
        target = parent[target]
    return route[::-1]


def _space_connected(body):
    """空格之间自身连通，没有被身体隔开的"死区"。

    不允许借道蛇头或尾格：两片空地只靠蛇头相连时，蛇头进入一侧就会
    把另一侧永久封死，之后也无法再把蛇身锁进哈密顿环。
    """
    free = set(ALL_CELLS).difference(body)
    if not free:
        return True
    start = next(iter(free))
    seen, stack = {start}, [start]
    while stack:
        for nb in _NEIGHBORS[stack.pop()]:
            if nb in free and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(free)


def _timed_route(snake, goal):
    """时间感知 BFS：第 k 节身体（0 为蛇头）在 len-k 步后离开原格。

    普通 BFS 把整条身体当墙；这里允许路线在蛇尾让出后穿过原来的身体，
    能找到更短的吃食路线。返回不含蛇头的路线，不可达时返回 None。
    """
    length = len(snake)
    free_at = {cell: length - k for k, cell in enumerate(snake)}
    parent = {snake[0]: None}
    frontier, t = [snake[0]], 0
    while frontier:
        t += 1
        next_frontier = []
        for cell in frontier:
            for nb in _NEIGHBORS[cell]:
                if nb in parent or free_at.get(nb, 0) > t:
                    continue
                parent[nb] = cell
                if nb == goal:
                    return _trace(parent, goal)
                next_frontier.append(nb)
        frontier = next_frontier
    return None


def _lock_hamiltonian_cycle(snake, budget=LOOKAHEAD_LOCK_BUDGET):
    """寻找"蛇头 → 全部空格 → 蛇尾"的哈密顿路径，把蛇身嵌进一条哈密顿环。

    成功时蛇身在环上恰好连续排在蛇头之后，正是 `_smart_safe_actions`
    的安全前提，此后沿环走必定铺满棋盘。DFS 按 Warnsdorff 规则先走
    出口最少的格子，并剪掉剩余空格不连通的分支。
    返回 (环索引或 None, 是否耗尽搜索预算)。
    """
    free = set(ALL_CELLS).difference(snake)
    ends = free.intersection(_NEIGHBORS[snake[-1]])
    path = []
    remaining = [budget]

    def exits(cell):
        return sum(nb in free for nb in _NEIGHBORS[cell])

    def still_connected():
        start = next(iter(free))
        seen, stack = {start}, [start]
        while stack:
            for nb in _NEIGHBORS[stack.pop()]:
                if nb in free and nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return len(seen) == len(free) and not ends.isdisjoint(seen)

    def extend(cell):
        if not free:
            return cell in ends
        remaining[0] -= 1
        if remaining[0] < 0 or not still_connected():
            return False
        for nb in sorted((nb for nb in _NEIGHBORS[cell] if nb in free), key=exits):
            free.remove(nb)
            path.append(nb)
            if extend(nb):
                return True
            free.add(nb)
            path.pop()
        return False

    if not ends or not extend(snake[0]):
        return None, remaining[0] < 0
    order = [snake[0], *path, *snake[:0:-1]]
    return {cell: i for i, cell in enumerate(order)}, False


class LookaheadPlanner:
    """预判规划器：先让"虚拟蛇"把路线试走一遍，再决定真蛇怎么走。

    1. 预判冲刺 sprint：时间感知 BFS 找最短吃食路线（考虑蛇尾逐步让路）；
       虚拟蛇吃到食物后必须仍能回到蛇尾，蛇长达到 LOOKAHEAD_CONNECT_LENGTH
       后还要求空白区域保持连通，避免把棋盘切碎；
    2. 追尾周旋 stall：直接去吃不安全时，选仍能回到蛇尾、且离尾巴最远的
       方向兜圈，等身体让出空间；
    3. 哈密顿锁定 locked：蛇长达到 LOOKAHEAD_LOCK_LENGTH 或长期吃不到时，
       把蛇身嵌进一条动态哈密顿环，之后沿环抄捷径，保证铺满棋盘。

    每一步要么保持"蛇头能回到蛇尾"，要么沿已验证的路线前进，所以不会
    撞墙或撞到自己。safe_actions 返回同样好的候选动作，由 DQN 最终拍板；
    routes 记录每个候选的预演路线，供界面画出 AI 的"思路"。
    """

    def __init__(self, lock=True):
        self.lock_enabled = lock
        self.reset()

    def reset(self):
        self.cycle = None          # 锁定后的哈密顿环索引
        self.phase = None          # 本步决策阶段：sprint / stall / locked / escape
        self.routes = {}           # 候选动作 → 预演路线（首格为下一步）
        self._memory = None        # 上一步所选吃食路线的剩余部分
        self._target = None
        self._last_score = 0
        self._last_meal = 0        # 最近一次吃到食物时的步数
        self._next_lock_try = 0

    def safe_actions(self, env):
        if env.food is None:
            self.routes = {}
            return []
        self._observe(env)
        if self.cycle is None:
            self._try_lock(env)
        if self.cycle is not None:
            self.phase = "locked"
            allowed = _smart_safe_actions(env, self.cycle)
            self.routes = _cycle_routes(env, self.cycle, allowed)
            return allowed
        return self._lookahead(env)

    def _observe(self, env):
        """记录吃食进度，并记住真蛇实际沿用的那条吃食路线。"""
        if env.score != self._last_score:
            self._last_score, self._last_meal = env.score, env.steps
        head = env.snake[0]
        taken = next((route for route in self.routes.values()
                      if route[0] == head), None)
        self._memory = taken[1:] if (self.phase == "sprint" and taken
                                     and len(taken) > 1
                                     and env.food == self._target) else None
        self._target = env.food

    def _try_lock(self, env):
        stalled = env.steps - self._last_meal >= LOOKAHEAD_STALL_STEPS
        if not self.lock_enabled or env.steps < self._next_lock_try or \
                (len(env.snake) < LOOKAHEAD_LOCK_LENGTH and not stalled):
            return
        self.cycle, exhausted = _lock_hamiltonian_cycle(env.snake)
        if exhausted:
            self._next_lock_try = env.steps + LOOKAHEAD_LOCK_RETRY

    def _pick(self, phase, scored):
        """scored: 动作 → (评分, 路线)；保留评分最高的全部动作作为候选。"""
        best = max(score for score, _ in scored.values())
        self.phase = phase
        self.routes = {action: route for action, (score, route) in scored.items()
                       if score == best}
        return list(self.routes)

    def _lookahead(self, env):
        snake, food = env.snake, env.food
        moves = _legal_moves(env)
        guard_space = len(snake) >= LOOKAHEAD_CONNECT_LENGTH
        stalled = env.steps - self._last_meal >= LOOKAHEAD_STALL_STEPS

        # 1. 预判冲刺：虚拟蛇走完整条吃食路线，检查吃完后的局面
        strict, loose = {}, {}
        for action, cell in moves.items():
            if cell == food:
                route = [cell]
            else:
                rest = _timed_route(_advance(snake, [cell], food), food)
                if rest is None:
                    continue
                route = [cell, *rest]
            final = _advance(snake, route, food)
            parent = _survey(final)
            if final[-1] not in parent:
                continue
            target = strict if not guard_space or _space_connected(final) \
                else loose
            target[action] = (-len(route), route)
        if self._memory:
            for action, cell in moves.items():
                if cell == self._memory[0] and action not in strict:
                    # 上一步已验证的路线依然可行，保证冲刺途中总有退路
                    strict[action] = (-len(self._memory), self._memory)
                    loose.pop(action, None)
        if strict or (stalled and loose):
            return self._pick("sprint", strict or loose)

        # 2. 追尾周旋：保持能回到蛇尾，优先不切碎空间、离尾巴更远
        chase = {}
        for action, cell in moves.items():
            after = _advance(snake, [cell], food)
            parent = _survey(after)
            if after[-1] in parent:
                tail_route = _trace(parent, after[-1])
                spacious = not guard_space or _space_connected(after)
                chase[action] = ((spacious, len(tail_route)), [cell, *tail_route])
        if chase:
            return self._pick("stall", chase)
        if loose:
            return self._pick("sprint", loose)

        # 3. 空间求生：理论上不会发生，作为最后的保险
        if not moves:
            self.phase, self.routes = "escape", {}
            return []
        return self._pick("escape", {
            action: (len(_survey(_advance(snake, [cell], food))), [cell])
            for action, cell in moves.items()})


EVAL_STRATEGIES = ("raw", "shield", "smart", "lookahead", "perfect")


def evaluate_agent(strategy="smart", games=20, max_steps=500, seed=0,
                   model_path=MODEL_PATH):
    """无窗口、固定食物随机种子的评测；比较模型、安全层和规划器。"""
    if strategy not in EVAL_STRATEGIES:
        raise ValueError("未知评测策略")
    if games <= 0 or max_steps <= 0:
        raise ValueError("games 和 max_steps 必须是正整数")
    agent = None
    if strategy != "perfect":
        agent = DQNAgent()
        agent.load(model_path)

    results = []
    for game in range(games):
        env = SnakeEnv(max_steps=max_steps, rng=random.Random(seed + game))
        idx, order = _cycle_aligned_with_snake(env)
        planner = LookaheadPlanner()
        while env.alive and not env.won:
            if strategy == "perfect":
                action = _select_hamiltonian_action(env, idx, order)
            else:
                allowed = None
                if strategy == "shield":
                    allowed = _hamiltonian_safe_actions(env, idx)
                elif strategy == "smart":
                    allowed = _smart_safe_actions(env, idx)
                elif strategy == "lookahead":
                    allowed = planner.safe_actions(env)
                if allowed == []:
                    env.alive = False
                    env.termination_reason = "trapped"
                    break
                action = agent.select_action(env._get_state(), training=False,
                                             allowed_actions=allowed)
            env.step(action)
        results.append({"seed": seed + game, "score": env.score,
                        "steps": env.steps, "reason": env.termination_reason})

    reasons = Counter(result["reason"] for result in results)
    total_food = sum(result["score"] for result in results)
    total_steps = sum(result["steps"] for result in results)
    return {
        "strategy": strategy, "games": games, "seed": seed,
        "max_steps": max_steps,
        "mean_score": total_food / games,
        "best_score": max(result["score"] for result in results),
        "mean_steps": total_steps / games,
        "steps_per_food": total_steps / total_food if total_food else None,
        "wins": reasons["win"],
        "collisions": reasons["wall"] + reasons["self"],
        "terminations": dict(reasons), "results": results,
    }


def _positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return number


def build_parser():
    parser = argparse.ArgumentParser(description="DQN 贪吃蛇教学项目")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train", help="训练并保存 DQN 模型")
    train_parser.add_argument("episodes", nargs="?", type=_positive_int, default=2000)
    train_parser.add_argument("--render", action="store_true", help="显示训练过程")
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--model", type=Path, default=MODEL_PATH)
    train_parser.add_argument("--progress", type=Path, default=PROGRESS_PATH)

    play_parser = subparsers.add_parser("play", help="观看已训练模型")
    play_parser.add_argument("--games", type=_positive_int, default=5)
    play_parser.add_argument("--max-steps", type=_positive_int, default=500)
    play_parser.add_argument("--model", type=Path, default=MODEL_PATH)
    play_parser.add_argument(
        "--planner", choices=("lookahead", "smart"), default="lookahead",
        help="lookahead 预判规划（默认）；smart 固定哈密顿环捷径")
    play_parser.add_argument(
        "--unsafe", action="store_true",
        help="关闭规划与安全层，仅使用 DQN 的单步动作掩码")

    subparsers.add_parser("manual", help="使用方向键手动游玩")

    perfect_parser = subparsers.add_parser("perfect", help="运行哈密顿环必胜策略")
    perfect_parser.add_argument("fps", nargs="?", type=_positive_int, default=120)
    perfect_parser.add_argument("--max-steps", type=_positive_int)

    evaluate_parser = subparsers.add_parser("evaluate", help="无窗口评测策略表现")
    evaluate_parser.add_argument("--strategy", default="all",
                                 choices=("all", *EVAL_STRATEGIES))
    evaluate_parser.add_argument("--games", type=_positive_int, default=20)
    evaluate_parser.add_argument("--max-steps", type=_positive_int, default=500)
    evaluate_parser.add_argument("--seed", type=int, default=0)
    evaluate_parser.add_argument("--model", type=Path, default=MODEL_PATH)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.mode == "train":
        train_agent(args.episodes, render=args.render, model_path=args.model,
                    progress_path=args.progress, seed=args.seed)
    elif args.mode == "play":
        play_with_model(args.model, games=args.games, max_steps=args.max_steps,
                        planner=None if args.unsafe else args.planner)
    elif args.mode == "manual":
        play_manual()
    elif args.mode == "perfect":
        play_perfect(fps=args.fps, max_steps=args.max_steps)
    else:
        strategies = EVAL_STRATEGIES if args.strategy == "all" else (args.strategy,)
        reports = [evaluate_agent(strategy, args.games, args.max_steps,
                                  args.seed, args.model) for strategy in strategies]
        print(json.dumps(reports, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
