import torch
import numpy as np
from src.tablut import TablutGame, Player
from src.utils import GameVisualizer, PlayerType
from ppo_trainer import TablutPPONetwork, select_action, TablutEnv, get_valid_action_mask

def ppo_agent_move(game: TablutGame, model, device="cuda", temperature=0.1):
    """Move callback for PPO agent"""
    # Create environment wrapper to use with the PPO agent
    env = TablutEnv()
    env.game = game  # Use the current game state
    
    # Get observation
    obs = env._get_observation()
    
    # Get valid action mask
    valid_mask = get_valid_action_mask(game)
    valid_mask_tensor = torch.BoolTensor(valid_mask).to(device)
    
    # Forward pass through network
    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, _ = model(state_tensor)
        
        # Apply temperature and masking
        policy_logits = policy_logits.squeeze(0) / temperature
        policy_logits[~valid_mask_tensor] = float('-inf')
        action_probs = torch.softmax(policy_logits, dim=0)
        
        # Get best action
        action = torch.argmax(action_probs).item()
    
    # Convert action to board coordinates
    from_pos = action // 81
    to_pos = action % 81
    from_row, from_col = divmod(from_pos, 9)
    to_row, to_col = divmod(to_pos, 9)
    
    # Execute the move
    success, _ = game.move_piece(from_row, from_col, to_row, to_col)
    
    if success:
        return f"PPO agent moved from ({from_row},{from_col}) to ({to_row},{to_col})"
    else:
        # This should not happen with a properly trained model and valid action masking
        return f"PPO agent attempted invalid move from ({from_row},{from_col}) to ({to_row},{to_col})"

def main():
    # Configuration (edit these values directly)
    model_path = r"model\ppo_white_20250407_225550\tablut_ppo_white_wr97_ep2700.pth"  # Check this path!
    play_as = "black"  # "white" or "black"
    temperature = 0.1  # Lower temperature = more optimal moves
    use_cpu = False  # Set to True to force CPU usage
    
    # Set up device
    device = "cpu" if use_cpu or not torch.cuda.is_available() else "cuda"
    print(f"Using device: {device}")
    
    # Create and load the model
    model = TablutPPONetwork(in_channels=6).to(device)  # Changed to 6 channels to match TablutEnv
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please update the model_path variable in the script.")
        return
    
    model.eval()
    
    # Create game and visualizer
    game = TablutGame()
    visualizer = GameVisualizer()
    
    # Set up player types based on user choice
    human_player = Player.WHITE if play_as.lower() == "white" else Player.BLACK
    ai_player = Player.BLACK if human_player == Player.WHITE else Player.WHITE
    
    # Register PPO agent callback
    agent_callback = lambda g: ppo_agent_move(g, model, device, temperature)
    if ai_player == Player.WHITE:
        game.set_move_callback(agent_callback, Player.WHITE)
        white_player_type = PlayerType.RL
        black_player_type = PlayerType.GUI
    else:
        game.set_move_callback(agent_callback, Player.BLACK)
        white_player_type = PlayerType.GUI
        black_player_type = PlayerType.RL
    
    print(f"You are playing as {human_player.value}")
    print(f"PPO agent is playing as {ai_player.value} with temperature {temperature}")
    print("Game starting...")
    
    # Start the game
    visualizer.run(game, white_player_type=white_player_type, black_player_type=black_player_type)

if __name__ == "__main__":
    main()
