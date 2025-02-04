from tablut import TablutGame
from utils import GameVisualizer

if __name__ == "__main__":
    game = TablutGame()
    visualizer = GameVisualizer()
    visualizer.run(game)