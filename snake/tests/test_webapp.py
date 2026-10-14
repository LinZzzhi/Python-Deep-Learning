import random
import unittest
from unittest.mock import AsyncMock, patch

from enhancedSnake import DQNAgent, SnakeEnv, _cycle_aligned_with_snake, _smart_safe_actions
from webapp import _Inbox, _frame, _new_game, _run_game


class WebGameTests(unittest.IsolatedAsyncioTestCase):
    async def test_web_matches_cli_with_planning_enabled_and_disabled(self):
        agent = DQNAgent()
        agent.load()
        for safety in (True, False):
            with self.subTest(safety=safety):
                expected_env = SnakeEnv(max_steps=100, rng=random.Random(0))
                idx, _ = _cycle_aligned_with_snake(expected_env)
                expected = [_frame(expected_env)]
                while expected_env.alive and not expected_env.won:
                    allowed = _smart_safe_actions(expected_env, idx) if safety else None
                    action = agent.select_action(expected_env._get_state(), training=False,
                                                 allowed_actions=allowed)
                    expected_env.step(action)
                    expected.append(_frame(expected_env))

                inbox = _Inbox()
                messages = []

                async def send_json(message):
                    messages.append(message)
                    if message["type"] == "end":
                        inbox.queue.put_nowait({"type": "stop"})

                socket = type("Socket", (), {})()
                socket.send_json = send_json

                def seeded_env(**kwargs):
                    return SnakeEnv(rng=random.Random(0), **kwargs)

                with patch("webapp.get_agent", return_value=agent), \
                     patch("webapp.SnakeEnv", side_effect=seeded_env), \
                     patch("webapp.asyncio.sleep", new=AsyncMock()):
                    await _run_game(socket, inbox, "play", 240, safety, 100)

                self.assertEqual(messages[:-1], expected)
                self.assertEqual(messages[-1]["type"], "end")
                self.assertEqual(messages[-1]["score"], expected_env.score)
                self.assertEqual(messages[-1]["reason"], expected_env.termination_reason)

    def test_perfect_and_manual_games_have_no_training_limit(self):
        for mode in ("perfect", "manual"):
            env, _, _ = _new_game(mode, safety=True, max_steps=1)
            self.assertIsNone(env.max_steps)


if __name__ == "__main__":
    unittest.main()
