import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from tablut import TablutGame
from utils import Player, Piece, PlayerType
from datetime import datetime

def board_to_state_tensor(board):
    """
    Convert board to state tensor with shape (6, 9, 9)
    Channels: [White pieces, Black pieces, King, Escape tiles, Camp tiles, Empty tiles]
    """
    # Initialize tensor with zeros
    state = np.zeros((6, 9, 9), dtype=np.float32)
    
    # Fill channels based on board state
    for row in range(9):
        for col in range(9):
            piece = board[row][col]
            
            # Channel 0: White pieces
            if piece == Piece.WHITE:
                state[0, row, col] = 1
                
            # Channel 1: Black pieces
            elif piece == Piece.BLACK:
                state[1, row, col] = 1
                
            # Channel 2: King
            elif piece == Piece.KING:
                state[2, row, col] = 1
                
            # Channel 3: Escape tiles
            elif piece == Piece.ESCAPE:
                state[3, row, col] = 1
                
            # Channel 4: Camp tiles
            elif piece == Piece.CAMP:
                state[4, row, col] = 1
                
            # Channel 5: Empty tiles
            elif piece == Piece.EMPTY:
                state[5, row, col] = 1
                
    return torch.FloatTensor(state)

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        # Input: (6, 9, 9) state representation
        # Convolutional layers
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 9 * 9, 128)
        
        # Output layer: 4 directions × 8 possible distances = 32 action values per position
        # Maximum distance = 8 (across the board)
        self.fc2 = nn.Linear(128, 32 * 9 * 9)
        
        # Activation function
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        
        # Flatten
        x = x.view(-1, 64 * 9 * 9)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Reshape to (batch_size, 32, 9, 9) for directions and distances at each position
        # The 32 channels represent 4 directions × 8 possible distances
        return x.view(-1, 32, 9, 9)

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done, is_opponent_move=False):
        # For opponent moves, we don't store transitions as training samples
        if not is_opponent_move:
            self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (torch.stack(states), 
                actions, 
                torch.FloatTensor(rewards), 
                torch.stack(next_states), 
                torch.FloatTensor(dones))
    
    def __len__(self):
        return len(self.buffer)

class TablutRLAgent:
    def __init__(self, player, device='cpu'):
        self.player = player  # WHITE or BLACK
        self.device = device
        
        # Initialize Q-networks
        self.q_network = DQN().to(device)
        self.target_network = DQN().to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Initialize optimizer with lower learning rate
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=5e-5)
        
        # Use a more reasonable learning rate schedule
        # Step once every 50 episodes instead of every step
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=50,     # Every 50 episodes
            gamma=0.9         # Reduce LR by 10% each time
        )
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer(capacity=50000)  # Larger buffer
        
        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 0.3  # Start with moderate exploration
        self.epsilon_min = 0.1  # Higher minimum exploration
        self.epsilon_decay = 0.999  # Slower decay
        self.batch_size = 128  # Larger batch size
        
        # Update target network based on steps, not episodes
        self.update_target_every = 1000  # Update every 1000 steps
        self.steps_count = 0
        
        # Create directory for saving models
        os.makedirs('models', exist_ok=True)

    def get_valid_actions(self, game):
        """Get all valid actions for current player"""
        valid_actions = []  # (from_row, from_col, to_row, to_col)
        
        for row in range(9):
            for col in range(9):
                piece = game.board[row][col]
                
                # Check if piece belongs to current player
                if (self.player == Player.WHITE and piece in [Piece.WHITE, Piece.KING]) or \
                   (self.player == Player.BLACK and piece == Piece.BLACK):
                    
                    # Get valid moves for this piece
                    valid_moves = game.get_valid_moves(row, col)
                    
                    # Add as valid actions
                    for to_row, to_col in valid_moves:
                        valid_actions.append((row, col, to_row, to_col))
        
        return valid_actions

    def action_to_index(self, action):
        """
        Convert action tuple to (direction, distance) index
        - Directions: 0=up, 1=right, 2=down, 3=left
        - Distances: 0-7 (number of squares moved - 1)
        """
        from_row, from_col, to_row, to_col = action
        
        # Determine direction
        if from_row > to_row:  # Moving up
            direction = 0
            distance = from_row - to_row - 1  # -1 because distance 0 means moving 1 square
        elif from_col < to_col:  # Moving right
            direction = 1
            distance = to_col - from_col - 1
        elif from_row < to_row:  # Moving down
            direction = 2
            distance = to_row - from_row - 1
        else:  # Moving left
            direction = 3
            distance = from_col - to_col - 1
        
        # Clamp distance to valid range (0-7)
        distance = min(7, max(0, distance))
        
        # Compute the channel index (0-31)
        channel_idx = direction * 8 + distance
        
        return from_row, from_col, channel_idx

    def select_action(self, game, training=True):
        """Select an action using epsilon-greedy policy"""
        valid_actions = self.get_valid_actions(game)
        
        if not valid_actions:
            return None  # No valid actions
        
        # With probability epsilon, select a random action
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)
        
        # Otherwise, select action with highest Q-value
        state = board_to_state_tensor(game.board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state).squeeze(0)  # Shape: [32, 9, 9]
        
        # Find best action among valid actions
        best_q_value = float('-inf')
        best_action = None
        
        for action in valid_actions:
            from_row, from_col, channel_idx = self.action_to_index(action)
            q_value = q_values[channel_idx, from_row, from_col].item()
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        return best_action

    def train(self):
        """Train the Q-network using a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Move tensors to the correct device
        states = states.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Convert actions to indices for easy access
        action_indices = []
        for action in actions:
            from_row, from_col, channel_idx = self.action_to_index(action)
            action_indices.append((channel_idx, from_row, from_col))
        
        # Compute current Q-values
        curr_q_values = self.q_network(states)
        curr_q = torch.zeros(self.batch_size, device=self.device)
        
        for i in range(self.batch_size):
            channel_idx, from_row, from_col = action_indices[i]
            curr_q[i] = curr_q_values[i, channel_idx, from_row, from_col]
        
        # Compute target Q-values with masking for illegal moves
        with torch.no_grad():
            next_q_values = self.target_network(next_states)  # shape [batch_size, 32, 9, 9]
            
            # For each state in the batch, mask illegal actions and take max of valid actions
            masked_next_q = []
            for i in range(self.batch_size):
                # Skip if the episode is done
                if dones[i] > 0.5:
                    masked_next_q.append(torch.tensor(0.0, device=self.device))
                    continue
                    
                # For our simplified MDP approach where we only store White->White transitions
                # (after Black's move), the player will always be White in next_states
                # This is because we only store transitions where it's White's turn in both
                # current_state and next_state
                
                # Create fresh game instance for this next state
                temp_game = TablutGame()
                
                # Set up the game state from the tensor
                self._tensor_to_board(next_states[i], temp_game.board)
                
                # Set player to White since we've structured our training to only consider 
                # board states from White's perspective
                temp_game.current_player = Player.WHITE
                
                # Get valid actions for White in this state
                valid_actions = self.get_valid_actions(temp_game)
                
                # If no valid actions, set Q-value to 0
                if not valid_actions:
                    masked_next_q.append(torch.tensor(0.0, device=self.device))
                    continue
                    
                # Create mask with -infinity for all positions
                mask = torch.full((32, 9, 9), float('-inf'), device=self.device)
                
                # Mark valid actions with 0
                for action in valid_actions:
                    from_row, from_col, channel_idx = self.action_to_index(action)
                    mask[channel_idx, from_row, from_col] = 0.0
                
                # Apply mask and get maximum Q-value
                masked_q = next_q_values[i] + mask
                masked_next_q.append(torch.max(masked_q.view(-1)))
            
            # Stack to create batch tensor
            next_q = torch.stack(masked_next_q)
            
            # Compute target Q-values
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update network
        loss = nn.MSELoss()(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def _tensor_to_board(self, state_tensor, board):
        """Convert a state tensor back to a board representation for action validation"""
        # For each position, determine the piece type from the tensor channels
        for row in range(9):
            for col in range(9):
                if state_tensor[0, row, col] > 0.5:  # White piece
                    board[row][col] = Piece.WHITE
                elif state_tensor[1, row, col] > 0.5:  # Black piece
                    board[row][col] = Piece.BLACK
                elif state_tensor[2, row, col] > 0.5:  # King
                    board[row][col] = Piece.KING
                elif state_tensor[3, row, col] > 0.5:  # Escape
                    board[row][col] = Piece.ESCAPE
                elif state_tensor[4, row, col] > 0.5:  # Camp
                    board[row][col] = Piece.CAMP
                else:  # Empty
                    board[row][col] = Piece.EMPTY

    def update_target_network(self):
        """Update the target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filename, folder_name="default"):
        """Save model weights to file"""
        # Create base models directory
        os.makedirs('models', exist_ok=True)
        
        # Create subdirectory for this specific run
        model_dir = os.path.join('models', folder_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model
        torch.save(self.q_network.state_dict(), os.path.join(model_dir, filename))

    def load_model(self, filename):
        """Load model weights from file"""
        self.q_network.load_state_dict(torch.load(f'models/{filename}', 
                                                 map_location=self.device))
        self.target_network.load_state_dict(self.q_network.state_dict())

def rl_agent_move(game, agent):
    """Callback function for RL agent to make a move"""
    # Select action
    action = agent.select_action(game)
    
    if action:
        from_row, from_col, to_row, to_col = action
        game.move_piece(from_row, from_col, to_row, to_col)
        return f"RL agent moved from ({from_row}, {from_col}) to ({to_row}, {to_col})"
    else:
        return "RL agent couldn't find a valid move"

def train_agent(white_agent=None, num_episodes=1000):
    """Train white agent against a random black opponent"""
    # Create a unique folder name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"training_run_{timestamp}"
    
    # Initialize white agent if not provided
    if white_agent is None:
        white_agent = TablutRLAgent(Player.WHITE)
    
    # Create random black opponent
    random_black = TablutRLAgent(Player.BLACK, device=white_agent.device)
    random_black.epsilon = 1.0  # Always choose random actions
    
    # Training stats
    white_wins = 0
    black_wins = 0
    draws = 0
    
    print("Training white agent against random black opponent...")
    for episode in range(num_episodes):
        # Create a new game
        game = TablutGame()
        
        # Main game loop
        total_reward = 0
        
        while not game.is_game_over():
            current_player = game.current_player
            
            if current_player == Player.WHITE:  # White's turn (our agent)
                # Store current state before white's move
                current_state = board_to_state_tensor(game.board)
                
                # Initialize reward with default value of 0
                reward = 0.0
                
                # Get action from white agent
                action = white_agent.select_action(game, training=True)
                
                if action is None:
                    # No valid moves - black wins
                    black_wins += 1
                    break
                
                # Execute white's action
                from_row, from_col, to_row, to_col = action
                game.move_piece(from_row, from_col, to_row, to_col)
                
                # Check if game is over after white's move
                if game.is_game_over():
                    winner = game.get_winner()
                    if winner == Player.WHITE:
                        reward = 1.0  # Win
                        white_wins += 1
                        
                        # Store transition with done=True
                        next_state = board_to_state_tensor(game.board)
                        white_agent.replay_buffer.add(current_state, action, reward, next_state, True)
                        total_reward += reward
                    elif winner == Player.BLACK:
                        # This can happen if White's move enables King capture
                        reward = -1.0  # Loss
                        black_wins += 1
                        
                        # Store transition with loss penalty
                        next_state = board_to_state_tensor(game.board)
                        white_agent.replay_buffer.add(current_state, action, reward, next_state, True)
                        total_reward += reward
                    else:
                        # Draw
                        reward = 0.0
                        draws += 1
                        
                        # Store transition with draw outcome
                        next_state = board_to_state_tensor(game.board)
                        white_agent.replay_buffer.add(current_state, action, reward, next_state, True)
                        total_reward += reward
                    continue  # Skip to next episode
                
                # Black's turn (random opponent)
                black_action = random_black.select_action(game, training=True)
                
                if black_action is None:
                    # No valid moves - white wins
                    white_wins += 1
                    
                    # Store transition with done=True and win reward
                    next_state = board_to_state_tensor(game.board)
                    white_agent.replay_buffer.add(current_state, action, 1.0, next_state, True)
                    total_reward += 1.0
                    break
                
                # Execute black's action
                b_from_row, b_from_col, b_to_row, b_to_col = black_action
                game.move_piece(b_from_row, b_from_col, b_to_row, b_to_col)
                
                # Get state after black's move (this is white's next decision point)
                next_state = board_to_state_tensor(game.board)
                
                # Check if game is over after black's move
                done = game.is_game_over()
                if done:
                    winner = game.get_winner()
                    if winner == Player.WHITE:
                        reward = 1.0  # Win
                        white_wins += 1
                    elif winner == Player.BLACK:
                        reward = -1.0  # Loss
                        black_wins += 1
                    else:
                        reward = 0.0  # Draw
                        draws += 1
                
                # Store the complete transition (white's action + black's response)
                white_agent.replay_buffer.add(current_state, action, reward, next_state, done)
                total_reward += reward
            
            # Train the agent after each step where we collected a transition
            if len(white_agent.replay_buffer) >= white_agent.batch_size:
                loss = white_agent.train()
            
            # Update step count
            white_agent.steps_count += 1
            
            # Update target network based on steps
            if white_agent.steps_count % white_agent.update_target_every == 0:
                white_agent.update_target_network()
                print(f"Target network updated at step {white_agent.steps_count}")
            
            # Decay epsilon after each step (this is fine, as exploration should decrease with experience)
            white_agent.decay_epsilon()
        
        # Update learning rate once per EPISODE, not per step
        white_agent.scheduler.step()
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"White Wins: {white_wins}, Black Wins: {black_wins}, Draws: {draws}")
            print(f"White Epsilon: {white_agent.epsilon:.4f}")
            print(f"Total Reward: {total_reward:.2f}")
            print("-" * 40)
        
        # Save model and evaluate periodically
        if (episode + 1) % 100 == 0:
            # Save model
            white_agent.save_model(f"white_agent_ep{episode+1}.pt", folder_name)
            
            # Evaluate against random opponent
            random_opponent = TablutRLAgent(Player.BLACK, device=white_agent.device)
            random_opponent.epsilon = 1.0
            win_rate = evaluate_agent(white_agent, random_opponent, num_games=10)
            
            print(f"\nEvaluation at episode {episode+1}:")
            print(f"White Agent vs Random Win Rate: {win_rate:.2f}")
            print("-" * 40)
    
    # Save final model
    white_agent.save_model("white_agent_final.pt", folder_name)
    
    print(f"Training completed. Model saved in: models/{folder_name}/")
    return white_agent, folder_name

def visualize_game(white_agent, black_agent):
    """Run a visual game between two agents"""
    from utils import GameVisualizer
    
    # Create a new game
    game = TablutGame()
    
    # Set up the game visualizer
    visualizer = GameVisualizer()
    
    # Lower exploration rates for demonstration
    white_agent.epsilon = 0.05
    black_agent.epsilon = 0.05
    
    # Set move callbacks for the agents
    def white_callback(g):
        action = white_agent.select_action(g, training=False)
        if action:
            from_row, from_col, to_row, to_col = action
            success, reason = g.move_piece(from_row, from_col, to_row, to_col)
            return f"White agent moved from ({from_row}, {from_col}) to ({to_row}, {to_col})"
        return "White agent couldn't find a valid move"

    def black_callback(g):
        action = black_agent.select_action(g, training=False)
        if action:
            from_row, from_col, to_row, to_col = action
            success, reason = g.move_piece(from_row, from_col, to_row, to_col)
            return f"Black agent moved from ({from_row}, {from_col}) to ({to_row}, {to_col})"
        return "Black agent couldn't find a valid move"
    
    game.set_move_callback(white_callback, Player.WHITE)
    game.set_move_callback(black_callback, Player.BLACK)
    
    # Run the game with the visualizer, specifying this is for visualization
    visualizer.run(game, 
                  white_player_type=PlayerType.RL, 
                  black_player_type=PlayerType.RL,
                  is_visualization=True)

def count_parameters(model):
    """Calculate the total number of parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate_agent(agent, opponent_agent, num_games=100):
    """Evaluate an agent against an opponent by playing multiple games"""
    # Ensure opponent is on the same device as the agent
    if opponent_agent.device != agent.device:
        opponent_agent.device = agent.device
        opponent_agent.q_network = opponent_agent.q_network.to(agent.device)
        opponent_agent.target_network = opponent_agent.target_network.to(agent.device)
    
    wins = 0
    for game_num in range(num_games):
        game = TablutGame()
        
        # Set move callbacks
        if agent.player == Player.WHITE:
            white_callback = lambda g: rl_agent_move(g, agent)
            black_callback = lambda g: rl_agent_move(g, opponent_agent)
        else:
            white_callback = lambda g: rl_agent_move(g, opponent_agent)
            black_callback = lambda g: rl_agent_move(g, agent)
            
        game.set_move_callback(white_callback, Player.WHITE)
        game.set_move_callback(black_callback, Player.BLACK)
        
        # Add a maximum move limit to prevent infinite loops
        max_moves = 200  # Reasonable limit for Tablut
        move_count = 0
        
        # Play until game over or max moves reached
        while not game.is_game_over() and move_count < max_moves:
            current_player = game.current_player
            move_result = ""
            
            if current_player == Player.WHITE:
                move_result = white_callback(game)
            else:
                move_result = black_callback(game)
                
            # Check if a valid move was made
            if "couldn't find a valid move" in move_result:
                break  # Exit if no valid moves
                
            move_count += 1
        
        # If max moves reached, consider it a draw
        if move_count >= max_moves:
            # print(f"Game {game_num+1} reached move limit - considered a draw")
            continue
        
        # Check winner
        winner = game.get_winner()
        if winner == agent.player:
            wins += 1
    
    return wins / num_games

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize white agent
    white_agent = TablutRLAgent(Player.WHITE, device=device)
    
    # Print model parameter count
    num_params = count_parameters(white_agent.q_network)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Train white agent
    print("\nStarting training...")
    white_agent, folder_name = train_agent(white_agent, num_episodes=2000)
    print("Training completed!")
    
    # Evaluate against random agent
    random_agent = TablutRLAgent(Player.BLACK)
    random_agent.epsilon = 1.0  # Always take random actions
    
    win_rate = evaluate_agent(white_agent, random_agent)
    print(f"White Agent Win Rate vs Random: {win_rate:.2f}")
    
    # Run a visual game against a random opponent
    print("\nRunning a visual game against random opponent...")
    random_opponent = TablutRLAgent(Player.BLACK, device=white_agent.device)
    random_opponent.epsilon = 1.0
    visualize_game(white_agent, random_opponent)

if __name__ == "__main__":
    main()

