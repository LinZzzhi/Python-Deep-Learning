import contextlib
import io
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from enhancedSnake import (
    ACTION_SIZE,
    DIRECTIONS,
    DOWN,
    LEFT,
    DirectionBuffer,
    DQNAgent,
    LookaheadPlanner,
    MODEL_PATH,
    REWARD_DEATH,
    REWARD_TIMEOUT,
    RIGHT,
    STATE_SIZE,
    TOTAL_CELLS,
    UP,
    SnakeEnv,
    _build_hamiltonian_cycle,
    _cycle_aligned_with_snake,
    _cycle_routes,
    _hamiltonian_safe_actions,
    _legal_moves,
    _lock_hamiltonian_cycle,
    _masked_max_q,
    _masked_double_q,
    _prepare_hamiltonian_snake,
    _select_hamiltonian_action,
    _smart_safe_actions,
    _space_connected,
    _timed_route,
    build_parser,
    evaluate_agent,
)


class SnakeEnvTests(unittest.TestCase):
    def test_reset_returns_normalized_state(self):
        state = SnakeEnv(rng=random.Random(1)).reset()

        self.assertEqual(state.shape, (STATE_SIZE,))
        self.assertEqual(state.dtype, np.float32)
        self.assertTrue(np.all(np.abs(state[:2]) < 1))
        self.assertEqual(state[6:].sum(), 1)

    def test_tail_cell_is_safe_when_not_eating(self):
        env = SnakeEnv(rng=random.Random(1))
        env.snake = [(1, 1), (2, 1), (2, 2), (1, 2)]
        env.direction = RIGHT
        env.food = (10, 10)

        self.assertEqual(env._is_danger(DOWN), 0)
        _, _, done = env.step(1)

        self.assertFalse(done)
        self.assertEqual(env.snake[0], (1, 2))
        self.assertEqual(len(env.snake), 4)

    def test_wall_collision_is_terminal(self):
        env = SnakeEnv(rng=random.Random(1))
        env.snake = [(0, 0), (0, 1), (0, 2)]
        env.direction = UP
        env.food = (5, 5)

        _, reward, done = env.step(0)

        self.assertTrue(done)
        self.assertFalse(env.alive)
        self.assertEqual(reward, REWARD_DEATH)

    def test_timeout_updates_environment_state(self):
        env = SnakeEnv(max_steps=1, rng=random.Random(1))

        _, reward, done = env.step(3)

        self.assertTrue(done)
        self.assertTrue(env.timed_out)
        self.assertFalse(env.alive)
        self.assertLessEqual(reward, REWARD_TIMEOUT + 1)

    def test_full_board_returns_terminal_state_without_food(self):
        env = SnakeEnv(max_steps=None, rng=random.Random(1))
        head, food, tail = (0, 0), (1, 0), (2, 0)
        middle = [
            (x, y)
            for x in range(20)
            for y in range(20)
            if (x, y) not in {head, food, tail}
        ]
        env.snake = [head, *middle, tail]
        env.direction = DOWN
        env.food = food

        state, _, done = env.step(3)

        self.assertTrue(done)
        self.assertTrue(env.won)
        self.assertIsNone(env.food)
        self.assertEqual(len(env.snake), TOTAL_CELLS)
        np.testing.assert_array_equal(state[:2], np.zeros(2, dtype=np.float32))

    def test_invalid_action_is_rejected(self):
        env = SnakeEnv(rng=random.Random(1))
        with self.assertRaises(ValueError):
            env.step(ACTION_SIZE)

    def test_direction_buffer_turns_twice_without_reversing(self):
        env = SnakeEnv(max_steps=None, rng=random.Random(1))
        env.food = (0, 0)
        keys = DirectionBuffer()

        self.assertFalse(keys.push(env, LEFT))      # 直接反向仍被拒绝
        self.assertTrue(keys.push(env, UP))
        self.assertTrue(keys.push(env, LEFT))       # 同一帧连按：相对 ↑ 合法
        env.step(keys.next_action(env))
        env.step(keys.next_action(env))

        self.assertTrue(env.alive)
        self.assertEqual(env.snake[:2], [(9, 9), (10, 9)])
        self.assertEqual(keys.next_action(env), DIRECTIONS.index(LEFT))


class AgentTests(unittest.TestCase):
    def test_agent_never_selects_invalid_reverse(self):
        agent = DQNAgent()
        state = SnakeEnv(rng=random.Random(1))._get_state()
        with torch.no_grad():
            for parameter in agent.policy_net.parameters():
                parameter.zero_()
            agent.policy_net.fc3.bias.copy_(torch.tensor([0., 1., 100., 2.]))

        self.assertEqual(agent.select_action(state, training=False), 3)
        for _ in range(50):
            self.assertNotEqual(agent.select_action(state, training=True), 2)

    def test_target_max_excludes_reverse_even_when_trapped(self):
        states = torch.zeros((2, STATE_SIZE))
        states[:, 9] = 1  # facing right; left is a no-op
        states[1, 2:6] = 1
        values = torch.tensor([[1., 2., 100., 4.], [1., 2., 100., 4.]])

        torch.testing.assert_close(_masked_max_q(values, states), torch.tensor([4., 4.]))

    def test_double_dqn_uses_policy_choice_and_target_value(self):
        states = torch.zeros((2, STATE_SIZE))
        states[:, 9] = 1
        states[1, 2] = 1  # up is dangerous in the second state
        policy = torch.tensor([[9., 2., 100., 4.], [9., 2., 100., 4.]])
        target = torch.tensor([[3., 90., 100., 7.], [3., 90., 100., 7.]])

        result = _masked_double_q(policy, target, states)

        torch.testing.assert_close(result, torch.tensor([3., 7.]))

    def test_learning_updates_weights_with_finite_loss(self):
        agent = DQNAgent()
        env = SnakeEnv(max_steps=1, rng=random.Random(1))
        state = env._get_state()
        next_state, reward, done = env.step(3)
        for _ in range(64):
            agent.memory.push(state, 3, reward, next_state, done)
        before = agent.policy_net.fc3.weight.detach().clone()

        loss = agent.learn()

        self.assertTrue(np.isfinite(loss))
        self.assertEqual(agent.learn_steps, 1)
        self.assertFalse(torch.equal(before, agent.policy_net.fc3.weight))

    def test_target_max_ignores_dangerous_actions(self):
        q_values = torch.tensor([[1.0, 100.0, 3.0, 4.0],
                                 [1.0, 2.0, 3.0, 4.0]])
        states = torch.zeros((2, STATE_SIZE))
        states[0, 3] = 1  # action 1 is dangerous
        states[1, 2:6] = 1  # trapped states fall back to all actions

        result = _masked_max_q(q_values, states)

        torch.testing.assert_close(result, torch.tensor([4.0, 4.0]))

    def test_bundled_model_is_compatible(self):
        agent = DQNAgent()
        agent.load(MODEL_PATH)

        self.assertEqual(agent.epsilon, 0)
        for policy, target in zip(agent.policy_net.parameters(),
                                  agent.target_net.parameters()):
            torch.testing.assert_close(policy, target)

    def test_model_save_is_atomic_and_loadable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "model.pth"
            agent = DQNAgent()
            agent.save(path)

            self.assertTrue(path.is_file())
            self.assertFalse(path.with_name(f".{path.name}.tmp").exists())
            DQNAgent().load(path)


class HamiltonianTests(unittest.TestCase):
    def test_planner_looks_beyond_manhattan_greedy_step(self):
        env = SnakeEnv(max_steps=None, rng=random.Random(0))
        env.food = (0, 0)
        idx, _ = _cycle_aligned_with_snake(env)

        self.assertEqual(_hamiltonian_safe_actions(env, idx), [1, 3])
        self.assertEqual(_smart_safe_actions(env, idx), [1])
        # Moving down initially goes away from food, but takes the shorter safe route.
        self.assertGreater(abs(10 - env.food[0]) + abs(11 - env.food[1]),
                           abs(10 - env.food[0]) + abs(10 - env.food[1]))

    def test_planner_advances_when_body_blocks_food_route(self):
        env = SnakeEnv(max_steps=None, rng=random.Random(0))
        env.snake = [(x, 16) for x in range(12, 7, -1)] \
            + [(8, y) for y in range(15, 7, -1)]
        env.direction = RIGHT
        env.food = (9, 12)
        idx, _ = _cycle_aligned_with_snake(env)

        self.assertGreater((idx[env.food] - idx[env.snake[0]]) % TOTAL_CELLS,
                           (idx[env.snake[-1]] - idx[env.snake[0]]) % TOTAL_CELLS)
        self.assertEqual(_hamiltonian_safe_actions(env, idx), [1, 3])
        self.assertEqual(_smart_safe_actions(env, idx), [1])

    def test_smart_model_fills_board_across_seeds(self):
        agent = DQNAgent()
        agent.load(MODEL_PATH)
        for seed in range(5):
            env = SnakeEnv(max_steps=200_000, rng=random.Random(seed))
            idx, _ = _cycle_aligned_with_snake(env)
            while env.alive and not env.won:
                allowed = _smart_safe_actions(env, idx)
                self.assertTrue(allowed, f"seed {seed} 没有规划动作")
                self.assertTrue(set(allowed) <= set(_hamiltonian_safe_actions(env, idx)))
                action = agent.select_action(env._get_state(), training=False,
                                             allowed_actions=allowed)
                env.step(action)
            self.assertTrue(env.won, f"seed {seed}: {env.termination_reason}")
            self.assertEqual(len(set(env.snake)), TOTAL_CELLS)
            self.assertEqual(env.score, TOTAL_CELLS - 3)

    def test_cycle_contains_every_cell_once_and_is_connected(self):
        idx, order = _build_hamiltonian_cycle()

        self.assertEqual(len(idx), TOTAL_CELLS)
        self.assertEqual(len(set(order)), TOTAL_CELLS)
        wrapped = order[1:] + order[:1]
        self.assertTrue(all(
            abs(first[0] - second[0]) + abs(first[1] - second[1]) == 1
            for first, second in zip(order, wrapped)
        ))

    def test_strategy_fills_the_board(self):
        env = SnakeEnv(max_steps=None, rng=random.Random(42))
        idx, order = _build_hamiltonian_cycle()
        _prepare_hamiltonian_snake(env, idx, order)

        for _ in range(200_000):
            action = _select_hamiltonian_action(env, idx, order)
            _, _, done = env.step(action)
            if done:
                break

        self.assertTrue(done)
        self.assertTrue(env.won)
        self.assertEqual(env.score, TOTAL_CELLS - 3)
        self.assertEqual(len(env.snake), TOTAL_CELLS)

    def test_model_safety_shield_survives_500_steps(self):
        agent = DQNAgent()
        agent.load(MODEL_PATH)

        for seed in range(100):
            env = SnakeEnv(max_steps=500, rng=random.Random(seed))
            state = env.reset()
            idx, order = _cycle_aligned_with_snake(env)
            while env.alive and not env.won:
                allowed = _hamiltonian_safe_actions(env, idx)
                self.assertTrue(allowed, f"seed {seed} 没有安全动作")
                action = agent.select_action(
                    state, training=False, allowed_actions=allowed)
                state, _, done = env.step(action)
                if done:
                    break

            self.assertNotIn(
                env.termination_reason, {"self", "wall", "trapped"},
                f"seed {seed} 在第 {env.steps} 步非安全终止",
            )


def _walk(env, route):
    """把路线逐格翻译成动作执行，返回途中是否一直存活。"""
    for cell in route:
        head = env.snake[0]
        env.step(DIRECTIONS.index((cell[0] - head[0], cell[1] - head[1])))
        if not env.alive:
            return False
    return True


class LookaheadTests(unittest.TestCase):
    def test_timed_route_goes_through_vacating_body(self):
        env = SnakeEnv(max_steps=None, rng=random.Random(0))
        env.snake = [(1, 2), (0, 2), (0, 1), (1, 1), (1, 0)]
        env.direction = RIGHT
        env.food = (0, 0)      # 四周都是身体，把身体当墙时根本不可达

        route = _timed_route(env.snake, env.food)

        self.assertEqual(len(route), 4)
        self.assertTrue(set(route) & set(env.snake))
        self.assertTrue(_walk(env, route))
        self.assertEqual(env.score, 1)

    def test_lock_builds_cycle_with_body_right_behind_head(self):
        env = SnakeEnv(max_steps=None, rng=random.Random(0))

        idx, exhausted = _lock_hamiltonian_cycle(env.snake)

        self.assertFalse(exhausted)
        order = sorted(idx, key=idx.get)
        self.assertEqual(len(set(order)), TOTAL_CELLS)
        self.assertTrue(all(
            abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
            for a, b in zip(order, order[1:] + order[:1])))
        for k, cell in enumerate(env.snake):
            self.assertEqual(idx[cell], -k % TOTAL_CELLS)
        self.assertTrue(_smart_safe_actions(env, idx))

    def test_lock_rejects_isolated_hole(self):
        snake = [(0, 1), (1, 1), (1, 0)]       # (0, 0) 被蛇身围成孤立空洞

        self.assertEqual(_lock_hamiltonian_cycle(snake), (None, False))
        self.assertFalse(_space_connected(snake))

    def test_lookahead_never_collides_and_stays_legal(self):
        agent = DQNAgent()
        agent.load(MODEL_PATH)
        for seed in range(30):
            env = SnakeEnv(max_steps=500, rng=random.Random(seed))
            planner = LookaheadPlanner()
            while env.alive and not env.won:
                allowed = planner.safe_actions(env)
                self.assertTrue(allowed, f"seed {seed} 没有候选动作")
                self.assertTrue(set(allowed) <= set(_legal_moves(env)))
                self.assertEqual(set(planner.routes), set(allowed))
                action = agent.select_action(env._get_state(), training=False,
                                             allowed_actions=allowed)
                self.assertEqual(planner.routes[action][0], _legal_moves(env)[action])
                env.step(action)
            self.assertEqual(env.termination_reason, "timeout", f"seed {seed}")

    def test_lookahead_outscores_cycle_planner_in_fixed_budget(self):
        lookahead = evaluate_agent("lookahead", games=10, max_steps=500)
        smart = evaluate_agent("smart", games=10, max_steps=500)

        self.assertEqual(lookahead["collisions"], 0)
        self.assertGreater(lookahead["mean_score"], smart["mean_score"] * 1.3)

    def test_lookahead_locks_and_fills_board(self):
        report = evaluate_agent("lookahead", games=1, max_steps=200_000, seed=9)

        self.assertEqual(report["wins"], 1, report["terminations"])
        self.assertEqual(report["results"][0]["score"], TOTAL_CELLS - 3)

    def test_cycle_routes_follow_planner_to_food(self):
        env = SnakeEnv(max_steps=None, rng=random.Random(0))
        env.food = (0, 0)
        idx, _ = _cycle_aligned_with_snake(env)
        allowed = _smart_safe_actions(env, idx)

        route = _cycle_routes(env, idx, allowed)[allowed[0]]

        self.assertEqual(route[-1], env.food)
        self.assertTrue(_walk(env, route))
        self.assertEqual(env.score, 1)


class CliTests(unittest.TestCase):
    def test_evaluation_defaults_compare_all_strategies(self):
        args = build_parser().parse_args(["evaluate"])
        self.assertEqual((args.strategy, args.games, args.max_steps, args.seed),
                         ("all", 20, 500, 0))

    def test_evaluation_is_reproducible_without_model_for_perfect(self):
        kwargs = dict(strategy="perfect", games=2, max_steps=50, seed=7,
                      model_path="missing-model.pth")
        report = evaluate_agent(**kwargs)

        self.assertEqual(report, evaluate_agent(**kwargs))
        self.assertEqual(report["collisions"], 0)
        self.assertEqual(report["terminations"], {"timeout": 2})
        self.assertEqual([result["seed"] for result in report["results"]], [7, 8])

    def test_train_arguments(self):
        args = build_parser().parse_args(["train", "10", "--seed", "7"])
        self.assertEqual((args.mode, args.episodes, args.seed), ("train", 10, 7))

    def test_play_uses_lookahead_planner_by_default(self):
        args = build_parser().parse_args(["play"])
        self.assertFalse(args.unsafe)
        self.assertEqual(args.planner, "lookahead")

    def test_non_positive_arguments_are_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                build_parser().parse_args(["play", "--games", "0"])


if __name__ == "__main__":
    unittest.main()
