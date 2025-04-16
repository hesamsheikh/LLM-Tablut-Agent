import ollama
import json
from typing import List, Tuple, Optional
from src.tablut import TablutGame, Player, Piece
from src.prompts import SYSTEM_PROMPT, MOVE_PROMPT
from src.utils import GameVisualizer, PlayerType

class LLMPlayer:
    def __init__(self, model_name: str = "gemma3:1b", temperature: float = 0.7):
        """Initialize LLM player with specified model and temperature"""
        self.model = model_name
        self.temperature = temperature
        self.message_history = []
        self.system_prompt = SYSTEM_PROMPT
        self.move_prompt = MOVE_PROMPT
        self.game = None
        
    def set_game(self, game: TablutGame):
        """Set the game instance for this player"""
        self.game = game
        
    def _board_to_prompt(self) -> str:
        """Convert current board state to a prompt for the LLM"""
        board_str = self.game._board_to_string()
        current_player = self.game.current_player.value
        move_count = self.game.move_count

        return self.format_move_prompt(board_str, current_player, move_count)

    def _extract_json_from_response(self, response_text: str) -> str:
        """Extract JSON from response that might be wrapped in markdown code blocks."""
        # Check if response is wrapped in code blocks
        if "```json" in response_text:
            # Extract content between ```json and ```
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                return response_text[start:end].strip()
        # If no code blocks, return as is
        return response_text.strip()

    def format_move_prompt(self, board_str, current_player, move_count):
        """Format the move prompt with current game state.
        
        Args:
            game (TablutGame): Current game state
            last_move (str, optional): Not used in current prompt
            captured_pieces (str, optional): Not used in current prompt
        
        Returns:
            str: Formatted move prompt
        """
        return self.move_prompt.format(
            board_str=board_str,
            current_player=current_player,
            move_count=move_count
        )
    
    def get_move(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Get next move from LLM"""
        if not self.game:
            raise ValueError("Game not set. Call set_game() first.")
        
        # Get and record opponent's last move if available
        opponent = Player.BLACK if self.game.current_player == Player.WHITE else Player.WHITE
        opponent_move = self.game.get_last_move(opponent)
        
        if opponent_move:
            from_pos = opponent_move['from']
            to_pos = opponent_move['to']
            piece_type = opponent_move['piece'].value
            
            # Add opponent move to message history
            opponent_move_msg = {
                "role": "system",
                "content": json.dumps({
                    "opponent_move": f"[{from_pos[0]},{from_pos[1]}] to [{to_pos[0]},{to_pos[1]}]",
                    "piece": piece_type,
                    "player": opponent.value
                }, indent=2)
            }
            
            # Add to history if not already there
            if not self.message_history or "opponent_move" not in self.message_history[-1]['content']:
                self.message_history.append(opponent_move_msg)
        
        # Generate the prompt with current board state
        prompt = self._board_to_prompt()
        
        try:
            # Create messages list including history
            messages = [
                {"role": "system", "content": self.system_prompt}
            ]
            
            # Add relevant history (last few exchanges to keep context focused)
            if len(self.message_history) > 0:
                relevant_history = self.message_history  # use full message history
                messages.extend(relevant_history)
            
            # Add current prompt (which includes the board state)
            messages.append({"role": "user", "content": prompt})
            
            # Get response from Ollama with temperature and history
            response = ollama.chat(
                model=self.model,
                messages=messages,
                options={"temperature": self.temperature}
            )
            
            # Extract and parse JSON response
            try:
                response_content = self._extract_json_from_response(response['message']['content'])
                move_data = json.loads(response_content)
                
                # Extract move coordinates
                from_pos = move_data['move']['from']
                to_pos = move_data['move']['to']
                reasoning = move_data['reasoning']
                
                # Add LLM response to history with reasoning
                self.message_history.append({
                    "role": "assistant",
                    "content": json.dumps({
                        "move": f"[{from_pos[0]},{from_pos[1]}] to [{to_pos[0]},{to_pos[1]}]",
                        "reasoning": reasoning
                    }, indent=2)
                })
                
                return (from_pos[0], from_pos[1]), (to_pos[0], to_pos[1])
                
            except json.JSONDecodeError as e:
                print(f"Error parsing LLM response as JSON: {e}")
                print(f"Raw response: {response['message']['content']}")
                return None
                
        except Exception as e:
            print(f"Error getting move from LLM: {e}")
            return None

# Create a global instance for each player color
white_player = None
black_player = None

def llm_move_callback(game: TablutGame) -> str:
    """Callback function for LLM player moves"""
    global white_player, black_player
    
    # Initialize the appropriate player if not already done
    if game.current_player == Player.WHITE:
        if white_player is None:
            white_player = LLMPlayer()
        player = white_player
    else:
        if black_player is None:
            black_player = LLMPlayer()
        player = black_player
    
    # Set the game instance
    player.set_game(game)
    
    # Get the move
    move = player.get_move()
    
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
        # Add more detailed invalid move information to history
        player.message_history.append({
            "role": "system",
            "content": json.dumps({
                "invalid_move": {
                    "from": [from_row, from_col],
                    "to": [to_row, to_col]
                },
                "error": error,
                "explanation": f"""The move from [{from_row},{from_col}] to [{to_row},{to_col}] is invalid because: {error}.
Please choose a different move with your own pieces."""
            }, indent=2)
        })
        return f"LLM attempted invalid move: {error}"

def play_game(llm_color: str = "BLACK", model_name: str = "gemma3:1b", temperature: float = 0.7):
    """
    Start a game with LLM playing as the specified color.
    
    Args:
        llm_color: "WHITE" or "BLACK" (default: "BLACK")
        model_name: Name of the LLM model to use (default: "gemma3:1b")
        temperature: Sampling temperature for LLM (default: 0.7)
    """
    game = TablutGame()
    
    # Convert string color to Player enum
    llm_player = Player.WHITE if llm_color.upper() == "WHITE" else Player.BLACK
    
    # Set up LLM as the specified player
    game.set_move_callback(llm_move_callback, llm_player)
    
    # Configure player types for visualizer
    white_type = PlayerType.LLM if llm_player == Player.WHITE else PlayerType.GUI
    black_type = PlayerType.LLM if llm_player == Player.BLACK else PlayerType.GUI
    
    # Initialize the LLM player with specified model and temperature
    global white_player, black_player
    if llm_player == Player.WHITE:
        white_player = LLMPlayer(model_name, temperature)
    else:
        black_player = LLMPlayer(model_name, temperature)
    
    # Run game with appropriate configuration
    visualizer = GameVisualizer()
    visualizer.run(game, white_player_type=white_type, black_player_type=black_type)

if __name__ == "__main__":
    # Hardcoded configuration values instead of argparse
    color = 'BLACK'  # Options: 'WHITE' or 'BLACK'
    model = 'gemma3:4b'
    temperature = 0.7
    
    print(f"Starting game with LLM ({model}) playing as {color} (temperature: {temperature})")
    play_game(llm_color=color, model_name=model, temperature=temperature)
