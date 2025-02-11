from tablut import TablutGame
from utils import GameVisualizer, Player, PlayerType

if __name__ == "__main__":
    game = TablutGame()
    visualizer = GameVisualizer()
    
    visualizer.run(game, white_player_type=PlayerType.GUI, black_player_type=PlayerType.GUI)