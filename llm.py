import ollama
from typing import List, Tuple, Optional
from src.tablut import TablutGame, Player, Piece

class LLMPlayer:
    def __init__(self, model_name: str = "mistral"):
        """Initialize LLM player with specified model"""
        self.model = model_name
        self.message_history = []
        
        # Configurable rules string
        self.rules = """
        Tablut Rules:
        1. The game is played on a 9x9 board
        2. White controls the King (K) and white pieces (W)
        3. Black controls the black pieces (B)
        4. The goal for White is to help the King escape to any corner tile
        5. The goal for Black is to capture the King
        6. Pieces move orthogonally (like rooks in chess)
        7. Pieces cannot jump over other pieces
        8. Captures occur by sandwiching enemy pieces between two of your pieces
        9. The King is captured by surrounding it on all four sides (or 3 sides if against the castle)
        10. Special tiles:
            - Castle (C): Center tile, only King can occupy it
            - Camps (X): Black pieces can move through these
            - Escape (E): Corner tiles where King can escape
        """
        
    def _board_to_prompt(self, game: TablutGame) -> str:
        """Convert current board state to a prompt for the LLM"""
        board_str = game._board_to_string()
        current_player = game.current_player.value
        
        prompt = f"""
        Current board state (9x9):
        {board_str}
        
        Legend:
        - W: White piece
        - B: Black piece
        - K: King
        - E: Escape tile
        - C: Castle
        - X: Camp
        - .: Empty space
        
        You are playing as {current_player}.
        Provide your next move in the format: from_row,from_col to_row,to_col
        For example: "4,4 to 4,7" moves a piece from position (4,4) to (4,7)
        
        What is your next move?
        """
        return prompt

    def get_move(self, game: TablutGame) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get next move from LLM"""
        prompt = self._board_to_prompt(game)
        
        # Add current state to history
        self.message_history.append({
            "role": "system",
            "content": f"Game state:\n{game._board_to_string()}"
        })
        
        try:
            # Get response from Ollama
            response = ollama.chat(model=self.model, messages=[
                {"role": "system", "content": self.rules},
                {"role": "user", "content": prompt}
            ])
            
            move_str = response['message']['content']
            
            # Add LLM response to history
            self.message_history.append({
                "role": "assistant",
                "content": move_str
            })
            
            # Parse move from response
            # Expected format: "from_row,from_col to_row,to_col"
            from_pos, to_pos = move_str.split(" to ")
            from_row, from_col = map(int, from_pos.strip().split(","))
            to_row, to_col = map(int, to_pos.strip().split(","))
            
            return (from_row, from_col), (to_row, to_col)
            
        except Exception as e:
            print(f"Error getting move from LLM: {e}")
            # Return None to indicate failure
            return None

def llm_move_callback(game: TablutGame) -> str:
    """Callback function for LLM player moves"""
    player = LLMPlayer()
    move = player.get_move(game)
    
    if move is None:
        return "LLM failed to provide a valid move"
        
    from_pos, to_pos = move
    from_row, from_col = from_pos
    to_row, to_col = to_pos
    
    # Try to make the move
    success, error = game.move_piece(from_row, from_col, to_row, to_col)
    
    if success:
        return f"LLM moved from ({from_row},{from_col}) to ({to_row},{to_col})"
    else:
        # Add invalid move to history
        player.message_history.append({
            "role": "system",
            "content": f"Invalid move: {error}"
        })
        return f"LLM attempted invalid move: {error}"

# Example usage:
if __name__ == "__main__":
    game = TablutGame()
    
    # Set up LLM as black player
    game.set_move_callback(llm_move_callback, Player.BLACK)
    
    # Run game with GUI for white player and LLM for black
    from src.utils import GameVisualizer, PlayerType
    visualizer = GameVisualizer()
    visualizer.run(game, white_player_type=PlayerType.GUI, black_player_type=PlayerType.LLM)
