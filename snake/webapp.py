"""Web 控制台：在浏览器里选择模式、观看并操作贪吃蛇。

用法：
    python webapp.py [--host 127.0.0.1] [--port 8333]

页面（web/index.html）提供以下模式：
    play    → AI 演示（DQN + 预判规划 / 哈密顿捷径 / 纯 DQN，可视化 AI 思维）
    manual  → 键盘手动（方向键 / WASD / 屏幕方向键，可开启 AI 教练提示）
    duel    → 人机对决（同一初始局面与食物序列，三档 AI 难度）
    perfect → 哈密顿环必胜策略
    train   → 训练监控（实时得分/损失曲线，可提前停止）

游戏逻辑完全复用 enhancedSnake.py：服务端以 render=False 无头推进
环境，每个 tick 把棋盘快照经 WebSocket 推给浏览器，由前端 canvas 绘制；
浏览器侧的按键和按钮则作为控制消息回传。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import random
import threading
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from enhancedSnake import (
    DIRECTIONS,
    GRID_HEIGHT,
    GRID_WIDTH,
    DirectionBuffer,
    DQNAgent,
    LookaheadPlanner,
    MAX_STEPS_PER_EPISODE,
    MODEL_PATH,
    PROGRESS_PATH,
    SnakeEnv,
    _advance,
    _build_hamiltonian_cycle,
    _cycle_aligned_with_snake,
    _cycle_routes,
    _legal_moves,
    _prepare_hamiltonian_snake,
    _select_hamiltonian_action,
    _smart_safe_actions,
    _survey,
    reflex_actions,
    train_agent,
)

WEB_PAGE = Path(__file__).resolve().parent / "web" / "index.html"
MAX_FPS = 240
MAX_PLAY_STEPS = 1_000_000
DIR_NAMES = ("up", "down", "left", "right")   # 与 DIRECTIONS 顺序一致
PLANNERS = ("lookahead", "smart", "off")
DUEL_LEVELS = {"rookie": "off", "steady": "smart", "master": "lookahead"}

app = FastAPI(title="贪吃蛇 Web 控制台")

_agent: DQNAgent | None = None


def get_agent() -> DQNAgent:
    """缓存已加载模型的智能体，避免每次开局重复读盘。"""
    global _agent
    if _agent is None:
        _agent = DQNAgent()
        _agent.load(MODEL_PATH)
    return _agent


def reset_agent_cache() -> None:
    """训练覆盖模型后调用，让下一次演示加载新权重。"""
    global _agent
    _agent = None


@app.get("/", response_class=HTMLResponse)
async def index():
    return WEB_PAGE.read_text(encoding="utf-8")


@app.get("/healthz")
async def healthz():
    return {"model": MODEL_PATH.is_file()}


def _frame(env: SnakeEnv, insight: dict | None = None) -> dict:
    """棋盘快照：前端据此用 canvas 重绘整局画面。"""
    frame = {
        "type": "frame",
        "snake": [list(cell) for cell in env.snake],
        "food": list(env.food) if env.food else None,
        "score": env.score,
        "steps": env.steps,
        "alive": env.alive,
        "won": env.won,
        "reason": env.termination_reason,
    }
    if insight is not None:
        frame["insight"] = insight
    return frame


def _clamp_fps(value) -> int:
    try:
        return min(max(int(value), 1), MAX_FPS)
    except (TypeError, ValueError):
        return 15


def _clamp_max_steps(value) -> int:
    try:
        return min(max(int(value), 1), MAX_PLAY_STEPS)
    except (TypeError, ValueError):
        return MAX_STEPS_PER_EPISODE


def _insight(phase, route=(), q=None, allowed=None, action=None, **extra) -> dict:
    """AI 思维快照：决策阶段、预演路线、Q 值、候选动作和最终动作。"""
    return {"phase": phase, "plan": [list(cell) for cell in route or ()],
            "q": q, "allowed": allowed, "action": action, **extra}


class _Pilot:
    """AI 驾驶员：规划器圈出候选动作，DQN 在候选中拍板，并记录思考过程。"""

    def __init__(self, env: SnakeEnv, planner: str, agent: DQNAgent | None = None,
                 cycle=None):
        self.planner = planner
        self.agent = agent
        self.lookahead = LookaheadPlanner() if planner == "lookahead" else None
        self.idx = self.order = None
        if planner == "perfect":
            self.idx, self.order = cycle
        elif planner == "smart":
            self.idx, self.order = _cycle_aligned_with_snake(env)

    def decide(self, env: SnakeEnv):
        """返回 (动作, 思维快照)；规划器找不到安全动作时动作为 None。"""
        if self.planner == "perfect":
            if not _smart_safe_actions(env, self.idx):
                return None, _insight("trapped")
            action = _select_hamiltonian_action(env, self.idx, self.order)
            route = _cycle_routes(env, self.idx, [action])[action]
            return action, _insight("cycle", route, action=action)

        state = env._get_state()
        routes = {}
        if self.planner == "lookahead":
            allowed = self.lookahead.safe_actions(env)
            routes, phase = self.lookahead.routes, self.lookahead.phase
        elif self.planner == "smart":
            allowed = _smart_safe_actions(env, self.idx)
            routes, phase = _cycle_routes(env, self.idx, allowed), "cycle"
        else:
            allowed, phase = reflex_actions(state), "reflex"
        if not allowed:
            return None, _insight("trapped")
        action = self.agent.select_action(state, training=False,
                                          allowed_actions=allowed)
        return action, _insight(phase, routes.get(action), self.agent.q_values(state),
                                allowed, action)


class _Coach:
    """手动模式的 AI 教练：给出推荐路线，并在玩家即将把自己困住时预警。"""

    def __init__(self, agent: DQNAgent | None):
        self.agent = agent
        self.planner = LookaheadPlanner(lock=False)   # 玩家不按环走，不能锁定

    def advise(self, env: SnakeEnv, keys: DirectionBuffer) -> dict:
        allowed = self.planner.safe_actions(env)
        state = env._get_state()
        q = self.agent.q_values(state) if self.agent else None
        if allowed and q:
            action = max(allowed, key=lambda a: q[a])
        else:
            action = allowed[0] if allowed else None
        heading = keys.peek_action(env)
        return _insight("coach", self.planner.routes.get(action), q, allowed, action,
                        heading=heading,
                        warning=heading not in allowed and not _move_is_safe(env, heading))


def _move_is_safe(env: SnakeEnv, action: int) -> bool:
    """这一步合法，并且走完之后蛇头仍能回到蛇尾。"""
    cell = _legal_moves(env).get(action)
    if cell is None:
        return False
    body = _advance(env.snake, [cell], env.food)
    return body[-1] in _survey(body)


class _SharedFood:
    """对决专用的食物序列：双方第 k 颗食物取自同一随机流，被身体占据时顺延。

    直接共用随机种子时，食物位置会随双方蛇身不同而迅速分叉；
    这里只要该格对双方都空闲，两边就会出现在同一位置。
    """

    def __init__(self, seed: int):
        self.seed = seed
        self.count = 0

    def choice(self, free):
        stream = random.Random(self.seed * 1_000_003 + self.count)
        self.count += 1
        free = set(free)
        while True:
            cell = (stream.randrange(GRID_WIDTH), stream.randrange(GRID_HEIGHT))
            if cell in free:
                return cell


class _SoloGame:
    """单人棋盘：AI 演示、手动挑战或完美策略。"""

    def __init__(self, mode: str, planner: str = "lookahead",
                 max_steps: int = MAX_STEPS_PER_EPISODE,
                 agent: DQNAgent | None = None):
        self.mode = mode
        self.env = SnakeEnv(render=False,
                            max_steps=max_steps if mode == "play" else None)
        self.keys = DirectionBuffer()
        self.pilot = self.coach = self.insight = None
        if mode == "perfect":
            idx, order = _build_hamiltonian_cycle()
            _prepare_hamiltonian_snake(self.env, idx, order)
            self.pilot = _Pilot(self.env, "perfect", cycle=(idx, order))
        elif mode == "play":
            self.pilot = _Pilot(self.env, planner, agent)
        else:
            self.coach = _Coach(agent)
            self.insight = self.coach.advise(self.env, self.keys)

    @property
    def done(self) -> bool:
        return not self.env.alive or self.env.won

    def key(self, name):
        if self.mode == "manual" and name in DIR_NAMES:
            self.keys.push(self.env, DIRECTIONS[DIR_NAMES.index(name)])

    def tick(self):
        if self.coach is not None:
            self.env.step(self.keys.next_action(self.env))
            self.insight = None if self.done else self.coach.advise(self.env, self.keys)
            return
        action, self.insight = self.pilot.decide(self.env)
        if action is None:                     # 与 play_with_model 行为一致
            self.env.alive = False
            self.env.termination_reason = "trapped"
        else:
            self.env.step(action)

    def frame(self) -> dict:
        return _frame(self.env, self.insight)

    def summary(self) -> dict:
        return {"type": "end", "reason": self.env.termination_reason,
                "score": self.env.score, "steps": self.env.steps,
                "won": self.env.won}


class _DuelGame:
    """人机对决：两块棋盘同步推进，步数用完或比分已无悬念时结束。"""

    def __init__(self, level: str, max_steps: int, agent: DQNAgent,
                 seed: int | None = None):
        self.level = level if level in DUEL_LEVELS else "master"
        self.seed = random.randrange(2 ** 31) if seed is None else seed
        self.max_steps = max_steps
        self.human = SnakeEnv(max_steps=max_steps, rng=_SharedFood(self.seed))
        self.ai = SnakeEnv(max_steps=max_steps, rng=_SharedFood(self.seed))
        self.keys = DirectionBuffer()
        self.pilot = _Pilot(self.ai, DUEL_LEVELS[self.level], agent)
        self.insight = None

    @staticmethod
    def _finished(env: SnakeEnv) -> bool:
        return not env.alive or env.won

    @property
    def done(self) -> bool:
        human_done, ai_done = self._finished(self.human), self._finished(self.ai)
        if human_done and ai_done:
            return True
        # 一方已结束且比分落后，另一方只会继续加分，胜负已定
        return (human_done and self.ai.score > self.human.score) or \
            (ai_done and self.human.score > self.ai.score)

    def key(self, name):
        if name in DIR_NAMES and not self._finished(self.human):
            self.keys.push(self.human, DIRECTIONS[DIR_NAMES.index(name)])

    def tick(self):
        if not self._finished(self.human):
            self.human.step(self.keys.next_action(self.human))
        if not self._finished(self.ai):
            action, self.insight = self.pilot.decide(self.ai)
            if action is None:
                self.ai.alive = False
                self.ai.termination_reason = "trapped"
            else:
                self.ai.step(action)

    def winner(self) -> str:
        if self.human.score == self.ai.score:
            return "draw"
        return "human" if self.human.score > self.ai.score else "ai"

    def frame(self) -> dict:
        return {"type": "duel", "level": self.level, "max_steps": self.max_steps,
                "human": _frame(self.human), "ai": _frame(self.ai, self.insight)}

    def summary(self) -> dict:
        return {"type": "duel_end", "winner": self.winner(), "level": self.level,
                "human": self.human.score, "ai": self.ai.score,
                "human_reason": self.human.termination_reason,
                "ai_reason": self.ai.termination_reason}


class _Inbox:
    """把 WebSocket 收到的控制消息转成异步队列，供游戏循环消费。"""

    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()

    async def receive_loop(self, ws: WebSocket):
        try:
            while True:
                self.queue.put_nowait(await ws.receive_json())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            self.queue.put_nowait({"type": "disconnect"})

    async def next(self) -> dict:
        return await self.queue.get()

    def drain(self):
        while not self.queue.empty():
            yield self.queue.get_nowait()


async def _run_session(ws: WebSocket, inbox: _Inbox, make_game, fps):
    """通用游戏循环：按帧率推进 make_game() 创建的棋盘，局间等待重新开始。"""
    game = make_game()
    cur_fps = fps

    async def send(msg) -> bool:
        try:
            await ws.send_json(msg)
            return True
        except (RuntimeError, WebSocketDisconnect):
            return False

    if not await send(game.frame()):
        return

    while True:                                # 一局一轮，轮间等待重新开始
        while not game.done:
            for msg in list(inbox.drain()):
                kind = msg.get("type")
                if kind in ("disconnect", "stop"):
                    return
                if kind == "restart":
                    game = make_game()
                    if not await send(game.frame()):
                        return
                elif kind == "fps":
                    cur_fps = _clamp_fps(msg.get("value"))
                elif kind == "key":
                    game.key(msg.get("dir"))

            game.tick()
            if not await send(game.frame()):
                return
            await asyncio.sleep(1 / cur_fps)

        if not await send(game.summary()):
            return
        while True:                            # 本局结束，等待重新开始/退出
            msg = await inbox.next()
            kind = msg.get("type")
            if kind in ("disconnect", "stop"):
                return
            if kind == "fps":
                cur_fps = _clamp_fps(msg.get("value"))
            elif kind == "restart":
                game = make_game()
                if not await send(game.frame()):
                    return
                break


async def _run_game(ws: WebSocket, inbox: _Inbox, mode, fps, planner="lookahead",
                    max_steps=MAX_STEPS_PER_EPISODE, level="master"):
    needs_model = mode == "duel" or (mode == "play" and planner in PLANNERS)
    if needs_model and not MODEL_PATH.is_file():
        await ws.send_json({"type": "error",
                            "message": "未找到 snake_model.pth，请先运行训练"})
        return
    agent = get_agent() if MODEL_PATH.is_file() and mode != "perfect" else None
    if mode == "duel":
        def make_game():
            return _DuelGame(level, max_steps, agent)
    else:
        def make_game():
            return _SoloGame(mode, planner, max_steps, agent)
    await _run_session(ws, inbox, make_game, fps)


async def _run_train(ws: WebSocket, inbox: _Inbox, episodes, seed,
                     model_path, progress_path):
    episodes = max(1, episodes)
    stop_flag = threading.Event()
    updates: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def progress_cb(episode, score, best, epsilon, loss):
        loop.call_soon_threadsafe(updates.put_nowait, {
            "type": "train", "episode": episode, "score": score,
            "best": best, "eps": epsilon, "loss": loss})
        return not stop_flag.is_set()          # 返回 False → 提前停止训练

    task = asyncio.create_task(asyncio.to_thread(
        train_agent, episodes, seed=seed, model_path=model_path,
        progress_path=progress_path, progress_cb=progress_cb))

    disconnected = False
    stopped = False
    while not (task.done() and updates.empty()):
        while not updates.empty():
            update = updates.get_nowait()
            if disconnected:
                continue
            try:
                await ws.send_json(update)
            except (RuntimeError, WebSocketDisconnect):
                disconnected = True
                stop_flag.set()
        for msg in list(inbox.drain()):
            kind = msg.get("type")
            if kind == "disconnect":
                disconnected = True
                stopped = True
                stop_flag.set()
            elif kind == "stop":
                stopped = True
                stop_flag.set()
        await asyncio.sleep(0.05)

    score_history, _ = await task
    if model_path == MODEL_PATH:
        reset_agent_cache()                    # 下次演示加载新训练的权重
    if disconnected:
        return
    await ws.send_json({"type": "train_end", "stopped": stopped,
                        "episodes": len(score_history),
                        "best": max(score_history, default=0),
                        "model": str(model_path)})


@app.websocket("/ws")
async def game_socket(ws: WebSocket, mode: str = "play", fps: int = 15,
                      planner: str = "lookahead", safety: bool = True,
                      level: str = "master", episodes: int = 200, seed: int = 42,
                      max_steps: int = MAX_STEPS_PER_EPISODE,
                      model: str = "", progress: str = ""):
    await ws.accept()
    inbox = _Inbox()
    receiver = asyncio.create_task(inbox.receive_loop(ws))
    if planner not in PLANNERS:
        planner = "lookahead"
    if not safety:                             # 兼容旧参数 safety=false
        planner = "off"
    try:
        if mode == "train":
            await _run_train(ws, inbox, episodes, seed,
                             Path(model) if model else MODEL_PATH,
                             Path(progress) if progress else PROGRESS_PATH)
        elif mode in ("play", "manual", "perfect", "duel"):
            await _run_game(ws, inbox, mode, _clamp_fps(fps), planner,
                            _clamp_max_steps(max_steps), level)
        else:
            await ws.send_json({"type": "error", "message": f"未知模式 {mode}"})
    finally:
        receiver.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await receiver
        with contextlib.suppress(RuntimeError, WebSocketDisconnect):
            await ws.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description="贪吃蛇 Web 控制台")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8333)
    args = parser.parse_args(argv)
    print(f"贪吃蛇 Web 控制台已启动：http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
