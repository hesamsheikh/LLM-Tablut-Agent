import json
import os
import yaml
import ollama
import requests
from dotenv import load_dotenv
from typing import Optional, Tuple

from src.tablut import TablutGame, Player
from src.prompts import SYSTEM_PROMPT, MOVE_PROMPT, FEW_SHOT_WHITE_WIN, FEW_SHOT_BLACK_WIN
from src.utils import GameVisualizer, PlayerType

# load up our environment stuff and config
load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)


class LLMPlayer:
    """this handles talking to the LLM to figure out moves in tablut"""
    def __init__(self, model_name: str = "gemma3:1b", temperature: float = 0.7, top_p: float = 0.3) -> None:
        self.model = model_name
        self.temperature = temperature
        self.top_p = top_p
        
        # check if we should use few-shot examples from the config
        provider_config = config.get(config.get('provider', 'ollama'), {})
        use_few_shot = provider_config.get('use_few_shot', False)
        
        current_system_prompt = SYSTEM_PROMPT
        few_shot_examples = ""
        if use_few_shot:
            # stick the examples before the main prompt
            few_shot_examples = f"{FEW_SHOT_WHITE_WIN}\n{FEW_SHOT_BLACK_WIN}\n"
        current_system_prompt = current_system_prompt.replace("{FEW_SHOT_EXAMPLES}", few_shot_examples)

        self.system_prompt = current_system_prompt
        self.move_prompt = MOVE_PROMPT
        self.reset()

    def reset(self) -> None:
        """clears everything - message history and game instance"""
        self.game = None
        self.message_history = [{"role": "system", "content": self.system_prompt}]

    def set_game(self, game: TablutGame) -> None:
        """link up a game instance with this player"""
        self.game = game

    def _format_board_with_tags(self, raw_board_str: str) -> str:
        """makes the board output look nice with proper formatting and tags"""
        lines = raw_board_str.strip().split('\n')
        header = "   0 1 2 3 4 5 6 7 8"
        separator = "  +-------------------+"
        formatted = ["<BOARD>", header, separator]
        for idx, line in enumerate(lines):
            cells = line.split()
            formatted.append(f"{idx} |{'|'.join(cells)}|")
        formatted.append("</BOARD>")
        return '\n'.join(formatted)

    def _board_to_prompt(self, opponent_move_str: str = "") -> str:
        """builds a prompt from the current board and what the opponent just did"""
        raw_board = self.game._board_to_string()
        formatted_board = self._format_board_with_tags(raw_board)
        current_player = self.game.current_player.value
        move_count = self.game.move_count
        return self.move_prompt.format(
            opponent_move=opponent_move_str,
            board_str=formatted_board,
            current_player=current_player,
            move_count=move_count
        )

    def _extract_json_from_response(self, response_text: str) -> str:
        """pulls out the JSON from the LLM response, gets rid of markdown wrapper stuff"""
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            if end != -1:
                return response_text[start:end].strip()
        return response_text.strip()

    def _prepare_prompt(self) -> str:
        """gets the prompt ready using current board state and what the other player just did"""
        opponent = Player.BLACK if self.game.current_player == Player.WHITE else Player.WHITE
        opponent_move = self.game.get_last_move(opponent)
        opponent_str = ""
        if opponent_move:
            from_pos = opponent_move['from']
            to_pos = opponent_move['to']
            opponent_str = f"I moved from [{from_pos[0]},{from_pos[1]}] to [{to_pos[0]},{to_pos[1]}]. "
            if "capture" in opponent_move:
                captured = opponent_move["capture"]
                opponent_str += f"I captured the piece at [{captured[0]},{captured[1]}]. "
        prompt = self._board_to_prompt(opponent_str)
        if prompt not in [msg['content'] for msg in self.message_history]:
            self.message_history.append({"role": "user", "content": prompt})
        return prompt

    def _call_llm(self) -> Optional[str]:
        """actually calls the LLM service and gets back the response"""
        try:
            if config.get('provider') == 'ollama':
                response = ollama.chat(
                    model=config['ollama']['model'],
                    messages=self.message_history,
                    options={"temperature": config['ollama']['temperature'], "top_p": config['ollama']['top_p']}
                )
                return self._extract_json_from_response(response['message']['content'])
            elif config.get('provider') == 'remote':
                api_key = os.getenv("DEEPSEEK_API_KEY")
                model_to_use = config['remote'].get('remote_model_override', config['remote']['model'])
                url = config['remote']['remote_api_url']
                payload = {
                    "model": model_to_use,
                    "messages": self.message_history,
                    "temperature": config['remote']['temperature'],
                    "top_p": config['remote']['top_p']
                }
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}"
                }
                resp = requests.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
                remote_response = data["choices"][0]["message"]["content"]
                return self._extract_json_from_response(remote_response)
            else:
                raise ValueError(f"Unknown provider: {config.get('provider')}")
        except Exception as ex:
            print(f"Error calling LLM: {ex}")
            return None

    def _parse_move_data(self, response_content: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int], str]]:
        """takes the JSON response from the LLM and pulls out the move coordinates and reasoning"""
        try:
            move_data = json.loads(response_content)
            from_pos = move_data['move']['from']
            to_pos = move_data['move']['to']
            reasoning = move_data['reasoning']
            return (from_pos[0], from_pos[1]), (to_pos[0], to_pos[1]), reasoning
        except json.JSONDecodeError as ex:
            print(f"Error parsing move data: {ex}")
            print(f"Raw response: {response_content}")
            return None

    def get_move(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """this is the main function - gets a move from the LLM by preparing the prompt, calling the service, and parsing what comes back"""
        if not self.game:
            raise ValueError("Game not set. Call set_game() first.")
        self._prepare_prompt()
        response_content = self._call_llm()
        if response_content is None:
            return None
        parsed = self._parse_move_data(response_content)
        if not parsed:
            return None
        from_pos, to_pos, reasoning = parsed
        self.message_history.append({
            "role": "assistant",
            "content": json.dumps({
                "move": {"from": [from_pos[0], from_pos[1]], "to": [to_pos[0], to_pos[1]]},
                "reasoning": reasoning
            }, indent=2)
        })
        return from_pos, to_pos


def llm_move_callback(game: TablutGame) -> str:
    """callback for LLM moves - creates a new player instance if we need one and makes the move"""
    # make a new player instance if this is the first move
    if not hasattr(game, 'llm_white') and game.current_player == Player.WHITE:
        game.llm_white = LLMPlayer()
    elif not hasattr(game, 'llm_black') and game.current_player == Player.BLACK:
        game.llm_black = LLMPlayer()

    # grab the right player
    player = game.llm_white if game.current_player == Player.WHITE else game.llm_black

    # hook up the game and make sure we have the initial board state
    player.set_game(game)
    if len(player.message_history) == 1:  # only system prompt exists
        prompt = player._board_to_prompt()
        player.message_history.append({"role": "user", "content": prompt})

    move = player.get_move()
    if move is None:
        return "LLM failed to provide a valid move"

    from_pos, to_pos = move
    from_row, from_col = from_pos
    to_row, to_col = to_pos
    success, error = game.move_piece(from_row, from_col, to_row, to_col)
    if success:
        return f"LLM moved from ({from_row},{from_col}) to ({to_row},{to_col})"
    else:
        player.message_history.append({
            "role": "user",
            "content": f"\nThe move you suggested is invalid:\n- From: [{from_row},{from_col}]\n- To: [{to_row},{to_col}]\nError: {error}\n"
        })
        return f"LLM attempted invalid move: {error}"


def play_game(llm_color: str = "BLACK", model_name: str = "gemma3:4b", temperature: float = 0.7, top_p: float = 0.3) -> None:
    """starts a game with the LLM playing against a human (GUI)"""
    game = TablutGame()
    target = Player.WHITE if llm_color.upper() == "WHITE" else Player.BLACK
    game.set_move_callback(llm_move_callback, target)

    visualizer = GameVisualizer()
    visualizer.run(game, white_player_type=PlayerType.LLM, black_player_type=PlayerType.GUI)


def play_game_llm_vs_ppo(llm_side: str = "BLACK",
                          llm_model_name: str = "gemma3:4b",
                          llm_temperature: float = 0.7,
                          llm_top_p: float = 0.3,
                          ppo_model_path: str = r"model\ppo_white_20250407_225550\tablut_ppo_white_wr97_ep2700.pth",
                          ppo_temperature: float = 0.1,
                          ppo_use_cpu: bool = False) -> None:
    """pit the LLM against a trained PPO agent - should be interesting!"""
    import torch
    from ppo_trainer import TablutPPONetwork, TablutEnv, get_valid_action_mask

    game = TablutGame()
    target = Player.WHITE if llm_side.upper() == "WHITE" else Player.BLACK
    
    device = "cpu" if ppo_use_cpu or not torch.cuda.is_available() else "cuda"
    ppo_model = TablutPPONetwork(in_channels=6).to(device)
    try:
        ppo_model.load_state_dict(torch.load(ppo_model_path, map_location=device))
        print(f"Loaded PPO model from {ppo_model_path}")
    except Exception as e:
        print(f"Error loading PPO model: {e}")
        return
    ppo_model.eval()

    def ppo_agent_move(game, model, device, temperature) -> str:
        env = TablutEnv()
        env.game = game
        obs = env._get_observation()
        valid_mask = get_valid_action_mask(game)
        valid_mask_tensor = torch.BoolTensor(valid_mask).to(device)
        state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            policy_logits, _ = model(state_tensor)
            policy_logits = policy_logits.squeeze(0) / temperature
            policy_logits[~valid_mask_tensor] = float('-inf')
            action = torch.argmax(torch.softmax(policy_logits, dim=0)).item()
        from_pos = action // 81
        to_pos = action % 81
        from_row, from_col = divmod(from_pos, 9)
        to_row, to_col = divmod(to_pos, 9)
        success, _ = game.move_piece(from_row, from_col, to_row, to_col)
        if success:
            return f"PPO agent moved from ({from_row},{from_col}) to ({to_row},{to_col})"
        else:
            return f"PPO agent attempted invalid move from ({from_row},{from_col}) to ({to_row},{to_col})"

    ppo_callback = lambda g: ppo_agent_move(g, ppo_model, device, ppo_temperature)

    if target == Player.WHITE:
        game.set_move_callback(llm_move_callback, Player.WHITE)
        game.set_move_callback(ppo_callback, Player.BLACK)
        white_type, black_type = PlayerType.LLM, PlayerType.RL
    else:
        game.set_move_callback(ppo_callback, Player.WHITE)
        game.set_move_callback(llm_move_callback, Player.BLACK)
        white_type, black_type = PlayerType.RL, PlayerType.LLM

    visualizer = GameVisualizer()
    visualizer.run(game, white_player_type=white_type, black_player_type=black_type)


def play_game_configurable() -> None:
    """starts a game based on whatever you put in the config file - pretty flexible"""
    game = TablutGame()
    white_conf = config.get('white_player', 'llm').lower()
    black_conf = config.get('black_player', 'gui').lower()

    # setup PPO stuff if we need it
    ppo_used = white_conf in ['ppo', 'rl'] or black_conf in ['ppo', 'rl']
    ppo_model = None
    if ppo_used:
        import torch
        from ppo_trainer import TablutPPONetwork, TablutEnv, get_valid_action_mask
        ppo_model_path = config.get('ppo_model_path', r"model\ppo_white_20250407_225550\tablut_ppo_white_wr97_ep2700.pth")
        ppo_temperature = config.get('ppo_temperature', 0.1)
        ppo_use_cpu = config.get('ppo_use_cpu', False)
        device = "cpu" if ppo_use_cpu or not torch.cuda.is_available() else "cuda"
        ppo_model = TablutPPONetwork(in_channels=6).to(device)
        try:
            ppo_model.load_state_dict(torch.load(ppo_model_path, map_location=device))
            print(f"Loaded PPO model from {ppo_model_path}")
        except Exception as ex:
            print(f"Error loading PPO model: {ex}")
            ppo_model = None
        if ppo_model:
            ppo_model.eval()

        def ppo_callback(game) -> str:
            if ppo_model is None:
                return "PPO model not available"
            from ppo_trainer import TablutEnv, get_valid_action_mask
            env = TablutEnv()
            env.game = game
            obs = env._get_observation()
            valid_mask = get_valid_action_mask(game)
            valid_mask_tensor = torch.BoolTensor(valid_mask).to(device)
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                policy_logits, _ = ppo_model(state_tensor)
                policy_logits = policy_logits.squeeze(0) / ppo_temperature
                policy_logits[~valid_mask_tensor] = float('-inf')
                action = torch.argmax(torch.softmax(policy_logits, dim=0)).item()
            from_pos = action // 81
            to_pos = action % 81
            from_row, from_col = divmod(from_pos, 9)
            to_row, to_col = divmod(to_pos, 9)
            success, _ = game.move_piece(from_row, from_col, to_row, to_col)
            if success:
                return f"PPO agent moved from ({from_row},{from_col}) to ({to_row},{to_col})"
            else:
                return f"PPO agent attempted invalid move from ({from_row},{from_col}) to ({to_row},{to_col})"
    
    from src.tablut import Player
    from src.utils import PlayerType
    if white_conf == "llm":
        game.set_move_callback(llm_move_callback, Player.WHITE)
        white_type = PlayerType.LLM
    elif white_conf in ["ppo", "rl"]:
        game.set_move_callback(ppo_callback, Player.WHITE)
        white_type = PlayerType.RL
    else:
        white_type = PlayerType.GUI

    if black_conf == "llm":
        game.set_move_callback(llm_move_callback, Player.BLACK)
        black_type = PlayerType.LLM
    elif black_conf in ["ppo", "rl"]:
        game.set_move_callback(ppo_callback, Player.BLACK)
        black_type = PlayerType.RL
    else:
        black_type = PlayerType.GUI

    visualizer = GameVisualizer()
    visualizer.run(game, white_player_type=white_type, black_player_type=black_type)


if __name__ == "__main__":
    print("Starting configurable game based on white_player and black_player settings.")
    play_game_configurable()
