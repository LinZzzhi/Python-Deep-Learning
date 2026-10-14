"""7.2 轻量演示入口：加载已训练模型即可观看 AI 自动游戏。"""

from enhancedSnake import play_with_model

if __name__ == "__main__":
    play_with_model(games=5, max_steps=500000)
