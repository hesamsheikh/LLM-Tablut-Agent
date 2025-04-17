import os
import json
import yaml

from llm import llm_move_callback
from src.tablut import TablutGame, Player
from src.utils import Piece


def random_move_callback(game):
    import random
    current_player = game.current_player
    valid_moves = []
    if current_player == Player.WHITE:
        for i in range(9):
            for j in range(9):
                piece = game.board[i][j]
                if piece in (Piece.WHITE, Piece.KING):
                    moves = game.get_valid_moves(i, j)
                    for move in moves:
                        valid_moves.append((i, j, move[0], move[1]))
    elif current_player == Player.BLACK:
        for i in range(9):
            for j in range(9):
                piece = game.board[i][j]
                if piece == Piece.BLACK:
                    moves = game.get_valid_moves(i, j)
                    for move in moves:
                        valid_moves.append((i, j, move[0], move[1]))
    if valid_moves:
        chosen_move = random.choice(valid_moves)
        r_from, c_from, r_to, c_to = chosen_move
        success, error = game.move_piece(r_from, c_from, r_to, c_to)
        if success:
            return f"Random agent moved from ({r_from},{c_from}) to ({r_to},{c_to})"
        else:
            return f"Random agent attempted invalid move from ({r_from},{c_from}) to ({r_to},{c_to})"
    else:
        return "Random agent found no valid moves"


def get_callback(player_config, side, config):
    if player_config.lower() == "llm":
        return llm_move_callback
    elif player_config.lower() == "ppo":
        import torch
        from ppo_trainer import TablutPPONetwork, TablutEnv, get_valid_action_mask
        device = "cuda" if torch.cuda.is_available() and not config["ppo_use_cpu"] else "cpu"
        ppo_model = TablutPPONetwork(in_channels=6).to(device)
        ppo_model.load_state_dict(torch.load(config["ppo_model_path"], map_location=device))
        ppo_model.eval()
        
        def ppo_callback(game):
            env = TablutEnv()
            env.game = game
            obs = env._get_observation()
            valid_mask = get_valid_action_mask(game)
            valid_mask_tensor = torch.BoolTensor(valid_mask).to(device)
            state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                policy_logits, _ = ppo_model(state_tensor)
                policy_logits = policy_logits.squeeze(0) / config["ppo_temperature"]
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
        
        return ppo_callback
    elif player_config.lower() == "random":
        return random_move_callback
    elif player_config.lower() == "gui":
        return force_valid_move
    else:
        raise ValueError(f"Unknown player type: {player_config}")


def force_valid_move(game):
    """
    Forces a valid move for the current player by scanning the board for a valid move.
    """
    cp = game.current_player
    for r in range(9):
        for c in range(9):
            piece = game.board[r][c]
            if cp == Player.WHITE and piece in (Piece.WHITE, Piece.KING):
                moves = game.get_valid_moves(r, c)
                if moves:
                    to_r, to_c = moves[0]
                    success, error = game.move_piece(r, c, to_r, to_c)
                    if success:
                        return f"Forced valid move: from ({r},{c}) to ({to_r},{to_c})"
            elif cp == Player.BLACK and piece == Piece.BLACK:
                moves = game.get_valid_moves(r, c)
                if moves:
                    to_r, to_c = moves[0]
                    success, error = game.move_piece(r, c, to_r, to_c)
                    if success:
                        return f"Forced valid move: from ({r},{c}) to ({to_r},{to_c})"
    return "No valid moves available"


def player_turn(game, callback, max_attempts=10):
    """
    Executes one turn for the current player using the provided callback, allowing multiple attempts until a valid move is made.
    Records invalid attempts and reasons.
    """
    invalid_count = 0
    reasons = {}
    for attempt in range(max_attempts):
        msg = callback(game)
        if msg.startswith("LLM moved") or msg.startswith("PPO agent moved") or msg.startswith("Random agent moved"):
            return msg, invalid_count, reasons
        else:
            invalid_count += 1
            if "attempted invalid move:" in msg:
                reason = msg.split("attempted invalid move:")[-1].strip()
            elif "failed to provide a valid move" in msg:
                reason = "failed to provide a valid move"
            else:
                reason = msg
            reasons[reason] = reasons.get(reason, 0) + 1
    forced_msg = force_valid_move(game)
    return forced_msg, invalid_count, reasons


def benchmark_game(config):
    """
    Plays a single game with players determined by config.
    It tracks the moves, counts of invalid moves, and reasons behind invalid moves per turn.
    """
    game = TablutGame()
    white_config = config["white_player"]
    black_config = config["black_player"]
    white_callback = get_callback(white_config, Player.WHITE, config)
    black_callback = get_callback(black_config, Player.BLACK, config)
    game.set_move_callback(white_callback, Player.WHITE)
    game.set_move_callback(black_callback, Player.BLACK)
    game_result = {"turns": [], "total_invalid_moves": 0, "invalid_reasons": {}}

    while not game.is_game_over():
        if game.current_player == Player.WHITE:
            msg, invalid_count, reasons = player_turn(game, game.white_move_callback)
        else:
            msg, invalid_count, reasons = player_turn(game, game.black_move_callback)
        
        print(f"Move executed: {msg} with {invalid_count} invalid moves")
        turn_info = {"move": msg, "invalid_moves": invalid_count, "reasons": reasons}
        game_result["turns"].append(turn_info)
        game_result["total_invalid_moves"] += invalid_count
        for reason, count in reasons.items():
            game_result["invalid_reasons"][reason] = game_result["invalid_reasons"].get(reason, 0) + count

    game_result["total_moves"] = game.move_count
    winner = game.get_winner()
    if winner == Player.WHITE:
        winning_player_type = white_config
    elif winner == Player.BLACK:
        winning_player_type = black_config
    else:
        winning_player_type = "Draw"
    game_result["winner"] = winning_player_type
    return game_result


def main():
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    num_games = config["num_games"]
    os.makedirs(os.path.join("logs", "benchmark"), exist_ok=True)

    all_games = []
    total_invalid_moves = 0
    all_invalid_reasons = {}
    llm_wins = 0

    for i in range(num_games):
        print(f"Starting game {i+1}...")
        game_result = benchmark_game(config)
        all_games.append(game_result)
        total_invalid_moves += game_result["total_invalid_moves"]
        for reason, count in game_result["invalid_reasons"].items():
            all_invalid_reasons[reason] = all_invalid_reasons.get(reason, 0) + count
        if game_result["winner"].lower() == "llm":
            llm_wins += 1

        # Save individual game result
        game_log_file = os.path.join("logs", "benchmark", f"game_{i+1}.json")
        with open(game_log_file, "w") as f:
            json.dump(game_result, f, indent=2)
        print(f"Game {i+1} completed: winner - {game_result['winner']}, moves - {game_result['total_moves']}, invalid moves - {game_result['total_invalid_moves']}")

    aggregate = {
        "total_games": num_games,
        "total_invalid_moves": total_invalid_moves,
        "average_invalid_moves": total_invalid_moves / num_games if num_games > 0 else 0,
        "invalid_reasons": all_invalid_reasons,
        "llm_win_rate": (llm_wins / num_games) * 100,
        "games": all_games
    }
    summary_log_file = os.path.join("logs", "benchmark", "benchmark_summary.json")
    with open(summary_log_file, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Benchmark completed for {num_games} games. Summary written to {summary_log_file}")


if __name__ == "__main__":
    main() 