import json
import os
import yaml
import ollama
import requests
from dotenv import load_dotenv
from typing import Optional, Tuple, List, Dict

from src.tablut import TablutGame, Player
from src.prompts import SYSTEM_PROMPT, SELECT_PIECE_PROMPT, SELECT_DESTINATION_PROMPT, FEW_SHOT_WHITE_WIN, FEW_SHOT_BLACK_WIN
from src.utils import GameVisualizer, PlayerType, Piece

# load up our environment and config
load_dotenv()
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

MAX_LLM_RETRIES = 3

class LLMTwoStepPlayer:
    """handles two-step interaction with LLM - first pick the piece, then pick where to move it"""
    def __init__(self, model_name: Optional[str] = None, temperature: Optional[float] = None, top_p: Optional[float] = None) -> None:
        provider = config.get('provider', 'ollama')
        provider_config = config.get(provider, {})

        self.model = model_name or provider_config.get('model')
        self.temperature = temperature if temperature is not None else provider_config.get('temperature', 0.7)
        self.top_p = top_p if top_p is not None else provider_config.get('top_p', 0.3)

        use_few_shot = provider_config.get('use_few_shot', False)
        current_system_prompt = SYSTEM_PROMPT
        few_shot_examples = ""
        if use_few_shot:
            few_shot_examples = f"{FEW_SHOT_WHITE_WIN}\n{FEW_SHOT_BLACK_WIN}\n"
        current_system_prompt = current_system_prompt.replace("{FEW_SHOT_EXAMPLES}", few_shot_examples)

        self.system_prompt = current_system_prompt
        self.select_piece_prompt_template = SELECT_PIECE_PROMPT
        self.select_destination_prompt_template = SELECT_DESTINATION_PROMPT
        self.reset()

    def reset(self) -> None:
        """clears everything and starts fresh"""
        self.game = None
        self.message_history = [{"role": "system", "content": self.system_prompt}]
        self.selected_piece_pos: Optional[Tuple[int, int]] = None

    def set_game(self, game: TablutGame) -> None:
        """hooks up a game instance"""
        self.game = game

    def _format_board_with_tags(self, raw_board_str: str) -> str:
        """makes the board look nice with proper formatting"""
        lines = raw_board_str.strip().split('\n')
        header = "   0 1 2 3 4 5 6 7 8"
        separator = "  +-------------------+"
        formatted = ["<BOARD>", header, separator]
        for idx, line in enumerate(lines):
            # board string should have piece values directly
            formatted_line = f"{idx} | {' '.join(line)} |" # adjust based on _board_to_string format
            formatted.append(formatted_line)
        formatted.append(separator) # add bottom separator
        formatted.append("</BOARD>")
        return '\n'.join(formatted)

    def _get_opponent_move_str(self) -> str:
        """figures out what the opponent just did so we can tell the LLM"""
        opponent = Player.BLACK if self.game.current_player == Player.WHITE else Player.WHITE
        opponent_move = self.game.get_last_move(opponent)
        opponent_str = ""
        if opponent_move:
            from_pos = opponent_move['from']
            to_pos = opponent_move['to']
            opponent_str = f"Opponent moved from [{from_pos[0]},{from_pos[1]}] to [{to_pos[0]},{to_pos[1]}]."
            if "capture" in opponent_move:
                captured = opponent_move["capture"]
                opponent_str += f" They captured the piece at [{captured[0]},{captured[1]}]."
        return opponent_str + "\n" if opponent_str else ""

    def _get_valid_sources(self) -> Dict[Tuple[int, int], List[Tuple[int, int]]]:
        """find all pieces we can move and where they can go"""
        valid_sources = {}
        current_p = self.game.current_player
        piece_types = (Piece.WHITE, Piece.KING) if current_p == Player.WHITE else (Piece.BLACK,)

        for r in range(9):
            for c in range(9):
                if self.game.board[r][c] in piece_types:
                    moves = self.game.get_valid_moves(r, c)
                    if moves:
                        valid_sources[(r, c)] = moves
        return valid_sources

    def _prepare_select_piece_prompt(self, valid_sources: List[Tuple[int, int]]) -> str:
        """builds the prompt for asking the LLM which piece to move"""
        raw_board = self.game._board_to_string()
        formatted_board = self._format_board_with_tags(raw_board)
        opponent_move_str = self._get_opponent_move_str()
        valid_sources_str = "\n".join([f"- [{r},{c}]" for r, c in valid_sources])

        prompt = self.select_piece_prompt_template.format(
            opponent_move=opponent_move_str,
            board_str=formatted_board,
            current_player=self.game.current_player.value,
            move_count=self.game.move_count,
            valid_sources_str=valid_sources_str
        )
        return prompt

    def _prepare_select_destination_prompt(self, selected_piece: Tuple[int, int], valid_destinations: List[Tuple[int, int]]) -> str:
        """builds the prompt for asking where to move the chosen piece"""
        raw_board = self.game._board_to_string()
        formatted_board = self._format_board_with_tags(raw_board)
        valid_destinations_str = "\n".join([f"- [{r},{c}]" for r, c in valid_destinations])

        prompt = self.select_destination_prompt_template.format(
            selected_piece_pos=f"[{selected_piece[0]},{selected_piece[1]}]",
            board_str=formatted_board,
            current_player=self.game.current_player.value,
            move_count=self.game.move_count,
            valid_destinations_str=valid_destinations_str
        )
        return prompt

    def _extract_json_from_response(self, response_text: str) -> str:
        """pulls JSON out of the LLM response, dealing with markdown wrapper nonsense"""
        # first, try finding the ```json ``` block
        try:
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end != -1:
                    potential_json = response_text[start:end].strip()
                    # make sure it's actually valid JSON
                    json.loads(potential_json)
                    return potential_json # return only if it parses
        except json.JSONDecodeError:
            # if ``` block exists but content isn't valid JSON, keep trying
            pass
        except Exception: # catch other potential errors during ``` processing
             pass

        # if no valid ```json block, try finding the outermost {}
        try:
            # strip whitespace that might mess things up
            stripped_text = response_text.strip()
            start = stripped_text.find('{')
            end = stripped_text.rfind('}')
            if start != -1 and end != -1 and end > start:
                potential_json = stripped_text[start:end+1]
                # make sure it's actually valid JSON
                json.loads(potential_json)
                return potential_json # return only if it parses
        except json.JSONDecodeError:
             # if {} found but content isn't valid JSON, keep going
            pass
        except Exception: # catch other potential errors during {} processing
            pass

        # fallback: return the original stripped text if we can't find valid JSON
        # this might allow the calling function's error handling to catch it
        print(f"Warning: Could not extract valid JSON from response: {response_text}")
        return response_text.strip()


    def _call_llm(self, prompt: str, step_history: List[Dict[str, str]]) -> Optional[str]:
        """actually calls the LLM with our prompt and history"""
        messages = self.message_history + step_history + [{"role": "user", "content": prompt}]
        try:
            if config.get('provider') == 'ollama':
                response = ollama.chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": self.temperature, "top_p": self.top_p}
                )
                return self._extract_json_from_response(response['message']['content'])
            elif config.get('provider') == 'remote':
                api_key = os.getenv("DEEPSEEK_API_KEY")
                model_to_use = config['remote'].get('remote_model_override', self.model)
                url = config['remote']['remote_api_url']
                payload = {
                    "model": model_to_use,
                    "messages": messages,
                    "temperature": self.temperature,
                    "top_p": self.top_p
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
            # print(f"Messages sent: {json.dumps(messages, indent=2)}") # debugging

    def _parse_select_piece_response(self, response_content: str) -> Optional[Tuple[Tuple[int, int], str]]:
        """tries to parse the piece selection response from the LLM"""
        data = None # keep data around for error logging
        try:
            # response_content comes from _extract_json_from_response
            data = json.loads(response_content) # if response_content is not valid JSON, this blows up
            if not isinstance(data, dict):
                 print(f"Error parsing piece selection: Expected a JSON object (dict), got {type(data)}")
                 print(f"Raw response content passed to json.loads: {response_content}")
                 return None

            piece_pos_list = data['selected_piece'] # this throws KeyError if 'selected_piece' is missing
            piece_pos = tuple(piece_pos_list)
            reasoning = data.get('reasoning', 'No reasoning provided.')

            if len(piece_pos) == 2 and all(isinstance(coord, int) for coord in piece_pos):
                 return (piece_pos[0], piece_pos[1]), reasoning
            else:
                print(f"Invalid coordinate format in piece selection: {piece_pos}")
                return None
        except json.JSONDecodeError as ex:
             print(f"Error decoding JSON for piece selection: {ex}")
             print(f"Raw response content: {response_content}")
             return None
        except KeyError as ex:
             print(f"Error parsing piece selection: Missing key {ex}")
             print(f"Parsed data: {data}")
             print(f"Raw response content: {response_content}")
             return None
        except (TypeError, IndexError) as ex: # catch issues like trying tuple() on non-iterable or accessing index
            print(f"Error processing parsed piece selection data: {ex}")
            print(f"Parsed data: {data}")
            print(f"Raw response content: {response_content}")
            return None

    def _parse_select_destination_response(self, response_content: str) -> Optional[Tuple[Tuple[int, int], str]]:
        """tries to parse the destination selection response from the LLM"""
        data = None # keep data around for error logging
        try:
            # response_content comes from _extract_json_from_response
            data = json.loads(response_content)
            if not isinstance(data, dict):
                 print(f"Error parsing destination selection: Expected a JSON object (dict), got {type(data)}")
                 print(f"Raw response content passed to json.loads: {response_content}")
                 return None

            dest_pos_list = data['selected_destination'] # KeyError possible here
            dest_pos = tuple(dest_pos_list)
            reasoning = data.get('reasoning', 'No reasoning provided.')

            if len(dest_pos) == 2 and all(isinstance(coord, int) for coord in dest_pos):
                return (dest_pos[0], dest_pos[1]), reasoning
            else:
                 print(f"Invalid coordinate format in destination selection: {dest_pos}")
                 return None
        except json.JSONDecodeError as ex:
             print(f"Error decoding JSON for destination selection: {ex}")
             print(f"Raw response content: {response_content}")
             return None
        except KeyError as ex:
             print(f"Error parsing destination selection: Missing key {ex}")
             print(f"Parsed data: {data}")
             print(f"Raw response content: {response_content}")
             return None
        except (TypeError, IndexError) as ex:
            print(f"Error processing parsed destination selection data: {ex}")
            print(f"Parsed data: {data}")
            print(f"Raw response content: {response_content}")
            return None


    def get_move(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """the main function - gets a move using two steps: first pick the piece, then pick where to move it
        this is more reliable than trying to get both decisions at once
        """
        if not self.game:
            raise ValueError("Game not set. Call set_game() first.")

        # keep track of any invalid attempts during this turn
        invalid_count = 0
        reasons = {}

        valid_sources_map = self._get_valid_sources()
        if not valid_sources_map:
            print("LLM Player: No valid source pieces found.")
            return None, invalid_count, reasons # no possible moves

        valid_source_coords = list(valid_sources_map.keys())

        # step 1: select piece
        selected_piece = None
        piece_reasoning = ""
        piece_history = []
        for attempt in range(MAX_LLM_RETRIES):
            prompt = self._prepare_select_piece_prompt(valid_source_coords)
            response_content = self._call_llm(prompt, piece_history)
            if response_content is None:
                piece_history.append({"role": "user", "content": prompt})
                piece_history.append({"role": "assistant", "content": "{\"error\": \"LLM call failed.\"}"})
                print(f"LLM Error (Piece Selection Attempt {attempt+1}/{MAX_LLM_RETRIES}): LLM call failed.")
                invalid_count += 1
                reasons["LLM call failed (piece selection)"] = reasons.get("LLM call failed (piece selection)", 0) + 1
                continue

            # try to parse what the LLM gave us
            parsed = self._parse_select_piece_response(response_content)
            if parsed:
                piece_pos, piece_reasoning = parsed
                if piece_pos in valid_source_coords:
                    selected_piece = piece_pos
                    # add successful interaction to step history
                    piece_history.append({"role": "user", "content": prompt})
                    piece_history.append({"role": "assistant", "content": response_content}) # log raw response associated with success
                    print(f"LLM selected piece: {selected_piece} (Attempt {attempt+1})")
                    # print(f"Reasoning: {piece_reasoning}") # optional: reduce noise?
                    break # valid piece selected
                else:
                    error_msg = f"Invalid piece selected: {piece_pos}. It's not in the list of valid pieces: {valid_source_coords}."
                    print(f"LLM Error (Piece Selection Attempt {attempt+1}/{MAX_LLM_RETRIES}): {error_msg}")
                    piece_history.append({"role": "user", "content": prompt})
                    # log the raw response that led to the error
                    piece_history.append({"role": "assistant", "content": response_content})
                    piece_history.append({"role": "user", "content": f"Error: {error_msg} Please choose a piece from the valid list provided."}) # ask for correction
                    invalid_count += 1
                    reasons[error_msg] = reasons.get(error_msg, 0) + 1
            else:
                 # parsing failed, _parse_select_piece_response already printed details
                 error_msg = "Failed to parse LLM response or validate data for piece selection."
                 print(f"LLM Error (Piece Selection Attempt {attempt+1}/{MAX_LLM_RETRIES}): {error_msg}")
                 piece_history.append({"role": "user", "content": prompt})
                 # log the raw response that led to the error
                 piece_history.append({"role": "assistant", "content": response_content})
                 piece_history.append({"role": "user", "content": f"Error: {error_msg} Please respond ONLY with the specified JSON format containing 'selected_piece' and 'reasoning'."}) # ask for correction
                 invalid_count += 1
                 reasons[error_msg] = reasons.get(error_msg, 0) + 1

        if selected_piece is None:
            print("LLM failed to select a valid piece after multiple attempts.")
            # persist the failed interaction history
            self.message_history.extend(piece_history)
            return None, invalid_count, reasons # return failure along with attempts info

        # persist the successful piece selection interaction history
        self.message_history.extend(piece_history)
        self.selected_piece_pos = selected_piece # store selected piece for potential future reference if needed

        # step 2: select destination
        valid_destinations = valid_sources_map[selected_piece]
        selected_destination = None
        dest_reasoning = ""
        dest_history = [] # separate history for this step, building on the main history
        for attempt in range(MAX_LLM_RETRIES):
            prompt = self._prepare_select_destination_prompt(selected_piece, valid_destinations)
            response_content = self._call_llm(prompt, dest_history) # use dest_history here
            if response_content is None:
                dest_history.append({"role": "user", "content": prompt})
                dest_history.append({"role": "assistant", "content": "{\"error\": \"LLM call failed.\"}"})
                print(f"LLM Error (Destination Selection Attempt {attempt+1}/{MAX_LLM_RETRIES}): LLM call failed.")
                invalid_count += 1
                reasons["LLM call failed (destination selection)"] = reasons.get("LLM call failed (destination selection)", 0) + 1
                continue

            # try to parse what the LLM gave us
            parsed = self._parse_select_destination_response(response_content)
            if parsed:
                dest_pos, dest_reasoning = parsed
                if dest_pos in valid_destinations:
                    selected_destination = dest_pos
                    # add successful interaction to step history
                    dest_history.append({"role": "user", "content": prompt})
                    # log raw response associated with success
                    dest_history.append({"role": "assistant", "content": response_content})
                    print(f"LLM selected destination: {selected_destination} (Attempt {attempt+1})")
                    # print(f"Reasoning: {dest_reasoning}") # optional: reduce noise?
                    break # valid destination selected
                else:
                    error_msg = f"Invalid destination selected: {dest_pos}. It's not in the list of valid destinations for piece {selected_piece}: {valid_destinations}."
                    print(f"LLM Error (Destination Selection Attempt {attempt+1}/{MAX_LLM_RETRIES}): {error_msg}")
                    dest_history.append({"role": "user", "content": prompt})
                    # log the raw response that led to the error
                    dest_history.append({"role": "assistant", "content": response_content})
                    dest_history.append({"role": "user", "content": f"Error: {error_msg} Please choose a destination from the valid list provided."}) # ask for correction
                    invalid_count += 1
                    reasons[error_msg] = reasons.get(error_msg, 0) + 1
            else:
                # parsing failed, _parse_select_destination_response already printed details
                error_msg = "Failed to parse LLM response or validate data for destination selection."
                print(f"LLM Error (Destination Selection Attempt {attempt+1}/{MAX_LLM_RETRIES}): {error_msg}")
                dest_history.append({"role": "user", "content": prompt})
                # log the raw response that led to the error
                dest_history.append({"role": "assistant", "content": response_content})
                dest_history.append({"role": "user", "content": f"Error: {error_msg} Please respond ONLY with the specified JSON format containing 'selected_destination' and 'reasoning'."}) # ask for correction
                invalid_count += 1
                reasons[error_msg] = reasons.get(error_msg, 0) + 1

        if selected_destination is None:
            print("LLM failed to select a valid destination after multiple attempts.")
            # persist the failed destination interaction history
            self.message_history.extend(dest_history)
            return None, invalid_count, reasons # return failure along with attempts info

        # persist the successful destination selection interaction history
        self.message_history.extend(dest_history)
        return (selected_piece, selected_destination), invalid_count, reasons # return success move with attempts info


def llm_two_step_move_callback(game: TablutGame) -> str:
    """callback for two-step LLM moves - creates the player if we need one and makes the move"""
    player_attr = f'llm_two_step_{game.current_player.value.lower()}'

    # make a new player instance if it doesn't exist for this player
    if not hasattr(game, player_attr):
        setattr(game, player_attr, LLMTwoStepPlayer())

    # get the right player instance
    player = getattr(game, player_attr)

    # hook up the game state for the player
    player.set_game(game)

    # make sure system prompt is the first message if history was reset or it's the first turn
    if not player.message_history or player.message_history[0]['role'] != 'system':
         player.reset() # resets history with system prompt
         player.set_game(game) # re-associate game

    move_result, invalid_count, reasons = player.get_move()

    if move_result is None:
        # LLM couldn't give us a valid move after trying several times
        # the get_move method already prints detailed errors
        fail_message = "LLM failed to provide a valid move after multiple attempts in two-step process (check logs for details)."
        # avoid adding duplicate error messages if get_move already added failure info
        if not player.message_history or "failed" not in player.message_history[-1].get("content", "").lower():
             player.message_history.append({"role": "user", "content": "Requesting move."}) # placeholder for the request attempt
             player.message_history.append({"role": "assistant", "content": f'{{\"error\": \"{fail_message}\"}}'})
        # return the failure message along with the accumulated invalid counts and reasons
        return fail_message, invalid_count, reasons

    from_pos, to_pos = move_result
    from_row, from_col = from_pos
    to_row, to_col = to_pos
    success, error = game.move_piece(from_row, from_col, to_row, to_col)

    if success:
        # everything worked - clean up if needed
        # player.selected_piece_pos = None
        success_message = f"LLM-2Step moved from ({from_row},{from_col}) to ({to_row},{to_col})"
        return success_message, invalid_count, reasons
    else:
        # this shouldn't happen often if get_move validates correctly against game.get_valid_moves,
        # but maybe some edge case in game.move_piece logic or concurrent state change? unlikely here
        error_msg = f"LLM-2Step final move rejected: from ({from_row},{from_col}) to ({to_row},{to_col}). Error: {error}" # more specific message
        # log this final critical failure exactly as requested
        # need the player *whose turn it was* when the move was attempted
        # since move_piece failed, current_player HAS NOT changed yet
        player_who_attempted = game.current_player
        # construct the reason key in the requested format
        final_error_reason = f"{player_who_attempted.value} attempted invalid move from ({from_row},{from_col}) to ({to_row},{to_col})"
        reasons[final_error_reason] = reasons.get(final_error_reason, 0) + 1
        invalid_count += 1 # count this final failure
        # add error context to LLM history for its next turn
        player.message_history.append({
            "role": "user",
            "content": f"The move you chose ({from_pos} -> {to_pos}) was ultimately invalid according to the game rules. Error: {error}. Please analyze the rules and board state again for your next turn."
        })
        return error_msg, invalid_count, reasons

# example of how to run a game (optional, primarily for testing)
# def play_game_llm_vs_gui(llm_color: str = "BLACK") -> None:
#     """starts a game with LLM (Two-Step) on one side and GUI on the other"""
#     game = TablutGame()
#     target = Player.WHITE if llm_color.upper() == "WHITE" else Player.BLACK
#     game.set_move_callback(llm_two_step_move_callback, target)

#     visualizer = GameVisualizer()
#     if target == Player.WHITE:
#         visualizer.run(game, white_player_type=PlayerType.LLM, black_player_type=PlayerType.GUI)
#     else:
#         visualizer.run(game, white_player_type=PlayerType.GUI, black_player_type=PlayerType.LLM)


# if __name__ == "__main__":
#     # example: LLM plays as Black against GUI White
#     play_game_llm_vs_gui(llm_color="BLACK") 