import ollama
import yaml
import os
from dotenv import load_dotenv
load_dotenv()
import openai
import json
from typing import List, Tuple, Optional
from src.tablut import TablutGame, Player, Piece
from src.prompts import SYSTEM_PROMPT, MOVE_PROMPT
from src.utils import GameVisualizer, PlayerType

# Load configuration for LLM settings from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

class LLMPlayer:
    def __init__(self, model_name: str = "gemma3:1b", temperature: float = 0.7, top_p: float = 0.3):
        """Initialize LLM player with specified model and temperature"""
        self.model = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = SYSTEM_PROMPT
        self.move_prompt = MOVE_PROMPT
        self.game = None
        # Initialize message history with system prompt
        self.message_history = [
            {"role": "system", "content": self.system_prompt}
        ]
        
    def set_game(self, game: TablutGame):
        """Set the game instance for this player"""
        self.game = game
        
    def _format_board_with_tags(self, raw_board_str: str) -> str:
        """Format raw board string into the <BOARD> tag format with row/column headers."""
        lines = raw_board_str.strip().split('\n')
        formatted_lines = ["<BOARD>"]
        # Add column numbers header
        formatted_lines.append("   0 1 2 3 4 5 6 7 8")
        formatted_lines.append("  +-------------------+")
        for row_num, line in enumerate(lines):
            cells = line.split()
            formatted_line = f"{row_num} |{'|'.join(cells)}|"
            formatted_lines.append(formatted_line)
        formatted_lines.append("</BOARD>")
        return '\n'.join(formatted_lines)

    def _board_to_prompt(self, opponent_move_str: str = "") -> str:
        """Convert current board state to a prompt for the LLM, including opponent's move if any"""
        raw_board_str = self.game._board_to_string()
        formatted_board = self._format_board_with_tags(raw_board_str)
        current_player = self.game.current_player.value
        move_count = self.game.move_count

        return self.format_move_prompt(
            board_str=formatted_board,
            current_player=current_player,
            move_count=move_count,
            opponent_move=opponent_move_str
        )

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

    def format_move_prompt(self, board_str, current_player, move_count, opponent_move=""):
        """Format the move prompt with current game state."""
        return self.move_prompt.format(
            opponent_move=opponent_move,
            board_str=board_str,
            current_player=current_player,
            move_count=move_count
        )
    
    def _prepare_prompt(self) -> str:
        # Prepare the prompt using the current board state and opponent's last move.
        opponent = Player.BLACK if self.game.current_player == Player.WHITE else Player.WHITE
        opponent_move = self.game.get_last_move(opponent)
        opponent_move_str = ""
        if opponent_move:
            from_pos = opponent_move['from']
            to_pos = opponent_move['to']
            opponent_move_str = f"I moved from [{from_pos[0]},{from_pos[1]}] to [{to_pos[0]},{to_pos[1]}]. "
        prompt = self._board_to_prompt(opponent_move_str)
        if prompt not in [m['content'] for m in self.message_history]:
            self.message_history.append({"role": "user", "content": prompt})
        return prompt

    def _call_llm(self) -> str:
        # Call the LLM service based on the configuration provider and return the extracted response content.
        try:
            if config.get('provider') == 'ollama':
                response = ollama.chat(
                    model=config['ollama']['model'],
                    messages=self.message_history,
                    options={"temperature": config['ollama']['temperature'], "top_p": config['ollama']['top_p']}
                )
                return self._extract_json_from_response(response['message']['content'])
            elif config.get('provider') == 'remote':
                openai.api_key = os.getenv("OPENAI_API_KEY")
                # Commented out the api_base assignment as per previous changes
                # openai.api_base = config['remote']['remote_api_url']
                model_to_use = config['remote'].get('remote_model_override', config['remote']['model'])
                response = openai.chat.completions.create(
                    model=model_to_use,
                    messages=self.message_history,
                    temperature=config['remote']['temperature'],
                    top_p=config['remote']['top_p']
                )
                remote_response = response.choices[0].message.content
                return self._extract_json_from_response(remote_response)
            else:
                raise ValueError(f"Unknown provider: {config.get('provider')}")
        except Exception as e:
            print(f"Error calling LLM: {e}")
            return None

    def _parse_move_data(self, response_content: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        # Parse the LLM JSON response to extract move coordinates and reasoning.
        try:
            move_data = json.loads(response_content)
            from_pos = move_data['move']['from']
            to_pos = move_data['move']['to']
            reasoning = move_data['reasoning']
            return (from_pos[0], from_pos[1]), (to_pos[0], to_pos[1]), reasoning
        except json.JSONDecodeError as e:
            print(f"Error parsing move data as JSON: {e}")
            print(f"Raw response: {response_content}")
            return None

    def get_move(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        if not self.game:
            raise ValueError("Game not set. Call set_game() first.")
        
        # Prepare the prompt and update message history
        self._prepare_prompt()

        # Call the LLM service to obtain the response
        response_content = self._call_llm()
        if response_content is None:
            return None
        
        # Parse the move data from the response
        parsed = self._parse_move_data(response_content)
        if not parsed:
            return None
        
        from_pos, to_pos, reasoning = parsed
        
        # Add the parsed LLM response (with reasoning) to the message history
        self.message_history.append({
            "role": "assistant",
            "content": json.dumps({
                "move": {
                    "from": [from_pos[0], from_pos[1]],
                    "to": [to_pos[0], to_pos[1]]
                },
                "reasoning": reasoning
            }, indent=2)
        })
        
        return from_pos, to_pos

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
        player.message_history.append(
            {
            "role": "user",
            "content": f"""
            The move you suggested is invalid:
            - From: [{from_row},{from_col}]
            - To: [{to_row},{to_col}]
            Error: {error}
            """
            }
        )
        return f"LLM attempted invalid move: {error}"

def play_game(llm_color: str = "BLACK", model_name: str = "gemma3:4b", temperature: float = 0.7, top_p: float = 0.3):
    """
    Start a game with LLM playing as the specified color.
    
    Args:
        llm_color: "WHITE" or "BLACK" (default: "BLACK")
        model_name: Name of the LLM model to use (default: "gemma3:4b")
        temperature: Sampling temperature for LLM (default: 0.7)
        top_p: Top-p sampling for LLM (default: 0.3)
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
        white_player = LLMPlayer(model_name, temperature, top_p)
    else:
        black_player = LLMPlayer(model_name, temperature, top_p)
    
    # Run game with appropriate configuration
    visualizer = GameVisualizer()
    visualizer.run(game, white_player_type=white_type, black_player_type=black_type)

if __name__ == "__main__":
    llm_color = config['llm_color']
    provider = config['provider']
    if provider == 'ollama':
        model = config['ollama']['model']
        temperature = config['ollama']['temperature']
        top_p = config['ollama']['top_p']
    elif provider == 'remote':
        model = config['remote']['model']
        temperature = config['remote']['temperature']
        top_p = config['remote']['top_p']
    else:
        raise ValueError(f"Unknown provider: {provider}")

    print(f"Starting game with LLM ({model}) playing as {llm_color} (temperature: {temperature} top_p: {top_p})")
    play_game(llm_color=llm_color, model_name=model, temperature=temperature, top_p=top_p)
