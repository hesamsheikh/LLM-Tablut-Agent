import os
import json
import yaml
import random # Keep random for the random player
from typing import Tuple

from llm import llm_move_callback
from llm_two_step import llm_two_step_move_callback # Import the new callback
from src.tablut import TablutGame, Player
from src.utils import Piece


def random_move_callback(game: TablutGame) -> str:
    """callback for a random player that just picks any valid move"""
    current_player = game.current_player
    valid_moves = []
    piece_types = (Piece.WHITE, Piece.KING) if current_player == Player.WHITE else (Piece.BLACK,)

    for r in range(9):
        for c in range(9):
            if game.board[r][c] in piece_types:
                moves = game.get_valid_moves(r, c)
                for move in moves:
                    valid_moves.append(((r, c), move)) # store as ((r_from, c_from), (r_to, c_to))

    if valid_moves:
        chosen_move = random.choice(valid_moves)
        (r_from, c_from), (r_to, c_to) = chosen_move
        success, error = game.move_piece(r_from, c_from, r_to, c_to)
        if success:
            return f"Random agent moved from ({r_from},{c_from}) to ({r_to},{c_to})", 0, {}
        else:
            # this really shouldn't happen if get_valid_moves is working right
            error_msg = f"Random agent attempted invalid move from ({r_from},{c_from}) to ({r_to},{c_to}). Error: {error}"
            return error_msg, 1, {error_msg: 1} # count this unlikely event as 1 invalid move
    else:
        fail_msg = "Random agent found no valid moves" # should only happen if game logic is stuck
        return fail_msg, 1, {fail_msg: 1} # count this as 1 invalid move


def get_callback(player_config: str, config: dict):
    """figure out which move callback to use based on player configuration"""
    player_type = player_config.lower()
    if player_type == "llm":
        return llm_move_callback
    elif player_type == "llm_two_step": # the new two-step type
        return llm_two_step_move_callback
    elif player_type == "random":
        return random_move_callback
    # removed PPO and GUI options as per request
    # elif player_type == "gui": # GUI is not suitable for automated benchmarking
    #     return None # or raise error
    else:
        raise ValueError(f"Unknown player type for benchmarking: {player_config}")


def player_turn(game: TablutGame, callback) -> Tuple[str, bool]:
    """
    runs one turn for the current player using the provided callback
    returns the final message, count of invalid LLM attempts, reasons for those attempts,
    and whether the move was actually executed by the game engine
    """
    if callback is None:
        return "No callback provided for player.", False

    # remember the state before trying to move
    initial_move_count = game.move_count
    initial_player = game.current_player
    initial_board_state = game._board_to_string() # for more robust checking

    # get result from callback (which now includes invalid attempt info)
    msg, invalid_count, reasons = callback(game)

    # check if the game state actually changed
    # a move is executed if the player changes, move count increases, or board state changes
    move_executed = (game.current_player != initial_player) or \
                    (game.move_count > initial_move_count) or \
                    (game._board_to_string() != initial_board_state) or \
                    game.is_game_over()

    # make sure the callback for random player also conforms (returns 0, {} for invalid count/reasons)
    # we might need to adjust random_move_callback if it doesn't return a tuple
    # let's assume for now callbacks ALWAYS return (str, int, dict)

    return msg, invalid_count, reasons, move_executed


def benchmark_game(config: dict):
    """
    runs a single game between two configured players without GUI
    keeps track of moves and failures
    """
    game = TablutGame()
    white_config = config["white_player"]
    black_config = config["black_player"]

    try:
        white_callback = get_callback(white_config, config)
        black_callback = get_callback(black_config, config)
    except ValueError as e:
        print(f"Configuration error: {e}")
        return {"error": str(e)}

    game.set_move_callback(white_callback, Player.WHITE)
    game.set_move_callback(black_callback, Player.BLACK)

    game_result = {
        "turns": [],
        "total_moves": 0,
        "winner": "Unknown",
        "reason": "Game incomplete"
    }

    MAX_TURNS = config.get("max_benchmark_turns", 100) # safety break

    while not game.is_game_over() and game.move_count < MAX_TURNS:
        current_player = game.current_player
        callback = game.white_move_callback if current_player == Player.WHITE else game.black_move_callback

        if callback is None: # should be caught by get_callback, but safety check
             print(f"Error: No callback set for {current_player.value}")
             game_result["error"] = f"Callback missing for {current_player.value}"
             break

        msg, invalid_count, reasons, move_executed = player_turn(game, callback)

        # turn_info = {"player": current_player.value, "message": msg, "executed": move_executed}
        # log in the desired format
        turn_info = {
            "player": current_player.value,
            "move": msg, # use the final message from the callback
            "invalid_moves": invalid_count,
            "reasons": reasons
        }
        game_result["turns"].append(turn_info)

        print(f"Turn {game.move_count+1} ({current_player.value}): {msg} (Internal Invalid: {invalid_count}, Executed: {move_executed})")

        if not move_executed:
            # decide how to handle failure: stop the game or force a random move?
            # for benchmarking, stopping might be better to measure reliability
            print(f"Player {current_player.value} failed to execute a move. Ending game.")
            game_result["reason"] = f"{current_player.value} failed to execute a valid move."
            break # stop game on failure

    game_result["total_moves"] = game.move_count
    winner = game.get_winner()

    if game.is_game_over():
         if winner == Player.WHITE:
             game_result["winner"] = white_config # store the configured player type
         elif winner == Player.BLACK:
             game_result["winner"] = black_config
         elif winner is None: # draw condition
             game_result["winner"] = "Draw"

         # figure out the reason for game end more accurately
         if game.move_count >= TablutGame.MOVE_LIMIT: # use class constant
             game_result["reason"] = f"Draw - Move limit ({TablutGame.MOVE_LIMIT} moves) reached"
         elif game.is_king_captured():
             game_result["reason"] = "Black wins - King captured"
         elif game.has_king_escaped():
              game_result["reason"] = "White wins - King escaped"
         # need to check state repetition and no valid moves from TablutGame if possible
         # elif game.state_count.get(game._board_to_string(), 0) >= 3: # assuming state_count accessible
         #      game_result["reason"] = "Draw - Repeated position"
         # elif not game._has_any_valid_moves(game.current_player): # assuming method accessible
         #      loser = game.current_player
         #      winning_player = Player.BLACK if loser == Player.WHITE else Player.WHITE
         #      game_result["reason"] = f"{winning_player.value} wins - {loser.value} has no valid moves"
         else:
              # default if other conditions met but specific reason unclear from benchmark script
              game_result["reason"] = f"Game ended. Winner: {game_result['winner']}"
    elif game.move_count >= MAX_TURNS:
        game_result["reason"] = f"Game stopped - Max benchmark turns ({MAX_TURNS}) reached."
        game_result["winner"] = "Incomplete"

    return game_result


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # make sure required benchmark config keys exist
    num_games = config.get("num_games", 1) # default to 1 game if not specified
    white_config = config.get("white_player", "random")
    black_config = config.get("black_player", "llm_two_step")
    eval_type = config.get("eval_type", "test") # for log folder naming

    # check if specified player types are valid for benchmarking
    try:
        get_callback(white_config, config)
        get_callback(black_config, config)
    except ValueError as e:
        print(f"Error in config.yaml player types: {e}")
        return

    save_dir = os.path.join("logs", "benchmark", f"w_{white_config}_vs_b_{black_config}_{eval_type}")
    os.makedirs(save_dir, exist_ok=True)

    # Find the highest existing game number to continue benchmarking runs
    existing_games = [f for f in os.listdir(save_dir) if f.startswith("game_") and f.endswith(".json")]
    start_game_num = 0
    if existing_games:
        try:
            highest_num = max(int(f.split("_")[1].split(".")[0]) for f in existing_games)
            start_game_num = highest_num
        except (ValueError, IndexError):
             print("Warning: Could not parse existing game numbers. Starting from 0.")
             start_game_num = 0

    all_game_results = []
    total_invalid_moves = 0
    all_invalid_reasons = {}
    wins = {white_config: 0, black_config: 0, "Draw": 0, "Incomplete": 0, "Unknown": 0} # Track wins by configured type

    print(f"Starting benchmark: {white_config} (White) vs {black_config} (Black)")
    print(f"Running {num_games} games. Saving results to: {save_dir}")
    print(f"Starting from game number {start_game_num + 1}")

    for i in range(num_games):
        current_game_num = start_game_num + i + 1
        print(f"\n--- Starting game {current_game_num} ---")
        game_result = benchmark_game(config)
        all_game_results.append(game_result)

        # Update aggregate stats from turn data
        for turn in game_result.get("turns", []):
            total_invalid_moves += turn.get("invalid_moves", 0)
            for reason, count in turn.get("reasons", {}).items():
                all_invalid_reasons[reason] = all_invalid_reasons.get(reason, 0) + count

        winner_type = game_result.get("winner", "Unknown")
        if winner_type in wins:
             wins[winner_type] += 1
        else:
             # Handle case where winner might be 'llm' but config was 'llm_two_step' etc.
             # This simple version assumes winner matches config key directly.
             wins["Unknown"] += 1


        # Save individual game result
        game_log_file = os.path.join(save_dir, f"game_{current_game_num}.json")
        try:
            with open(game_log_file, "w") as f:
                json.dump(game_result, f, indent=2)
            print(f"Game {current_game_num} completed: Winner - {game_result.get('winner', 'N/A')}, Reason - {game_result.get('reason', 'N/A')}, Moves - {game_result.get('total_moves', 0)}")
        except TypeError as e:
             print(f"Error saving game {current_game_num} log: {e}. Result was: {game_result}")


    # Calculate aggregate statistics
    total_games_run = len(all_game_results)
    win_rates = {player: (count / total_games_run) * 100 if total_games_run > 0 else 0
                 for player, count in wins.items()}

    aggregate_summary = {
        "config": {
            "white_player": white_config,
            "black_player": black_config,
            "num_games_requested": num_games,
            "eval_type": eval_type,
            "provider_settings": config.get(config.get('provider', 'ollama')), # Log LLM settings used
        },
        "results": {
            "total_games_completed": total_games_run,
            "wins": wins,
            "win_rates_percent": win_rates,
            "total_internal_invalid_moves": total_invalid_moves,
            "average_internal_invalid_moves": total_invalid_moves / total_games_run if total_games_run > 0 else 0,
            "internal_invalid_reasons": all_invalid_reasons,
        },
        # Optionally include all game details, but can make the summary file large
        # "games": all_game_results
    }

    summary_log_file = os.path.join(save_dir, "benchmark_summary.json")
    with open(summary_log_file, "w") as f:
        json.dump(aggregate_summary, f, indent=2)

    print(f"\n--- Benchmark Summary ---")
    print(f"Saved to: {summary_log_file}")
    print(f"Total Games Run: {total_games_run}")
    print(f"Win Counts: {wins}")
    print(f"Win Rates (%): { {p: f'{r:.2f}' for p, r in win_rates.items()} }")
    print(f"Total Internal Invalid Moves (across all turns): {total_invalid_moves}")
    print(f"Avg Internal Invalid Moves per Game: {aggregate_summary['results']['average_internal_invalid_moves']:.2f}")


if __name__ == "__main__":
    main() 