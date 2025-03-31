import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import copy
from typing import Tuple, List, Dict, Any, Optional
from utils import Piece, Player, PlayerType, GameVisualizer
from tablut import TablutGame

class DQN(nn.Module):
    def __init__(self, input_channels=6, board_size=9, action_dim=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, action_dim, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        return x

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        if len(self.buffer) < batch_size:
            return None
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class TablutRLAgent:
    # Default reward values stored in a dictionary for easy adjustment
    DEFAULT_REWARDS = {
        'win': 3.0,           # Winning the game
        'lose': -3.0,         # Losing the game
        'draw': 0.0,          # Game ends in a draw
        'move': -0.1,         # Small penalty for each move to encourage efficiency
        'capture_piece': 1.0, # Capturing opponent's piece
        'lose_piece': -1.0,   # Losing a piece
        'king_capture': 2.0,  # Black bonus for capturing king
        'king_escape': 3.0,   # White bonus for king reaching escape tile
        'king_approach_escape': 0.2, # White bonus for king moving toward escape
    }
    
    def __init__(self, player: Player, device='cuda' if torch.cuda.is_available() else 'cpu', rewards=None):
        self.player = player
        self.device = device
        
        # Set reward values (use defaults or custom values if provided)
        self.rewards = self.DEFAULT_REWARDS.copy()
        if rewards:
            self.rewards.update(rewards)
        
        # DQN networks
        self.q_network = DQN().to(device)
        self.target_network = copy.deepcopy(self.q_network).to(device)
        self.target_network.eval()
        
        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        self.target_update_freq = 10
        self.eval_epsilon = 0.05  # Epsilon to use during actual gameplay
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training stats
        self.training_step = 0
        self.episodes_played = 0
        self.wins = 0
        self.losses = 0
        self.draws = 0
    
    def board_to_state(self, game: TablutGame) -> torch.Tensor:
        """Convert board state to tensor representation"""
        # Create 6 channels: white, black, king, escape, camp, empty
        state = np.zeros((6, 9, 9), dtype=np.float32)
        
        for row in range(9):
            for col in range(9):
                piece = game.board[row][col]
                if piece == Piece.WHITE:
                    state[0, row, col] = 1.0
                elif piece == Piece.BLACK:
                    state[1, row, col] = 1.0
                elif piece == Piece.KING:
                    state[2, row, col] = 1.0
                elif piece == Piece.ESCAPE:
                    state[3, row, col] = 1.0
                elif piece == Piece.CAMP:
                    state[4, row, col] = 1.0
                elif piece == Piece.EMPTY:
                    state[5, row, col] = 1.0
                # CASTLE is represented implicitly by all zeroes
        
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def find_king_position(self, board) -> Optional[Tuple[int, int]]:
        """Helper method to find the king's position on the board"""
        for row in range(9):
            for col in range(9):
                if board[row][col] == Piece.KING:
                    return (row, col)
        return None
    
    def get_valid_actions_mask(self, game: TablutGame) -> np.ndarray:
        """Create a mask of valid actions for the current player"""
        # Create a mask for all possible actions (9x9x4)
        # 0: up, 1: right, 2: down, 3: left
        action_mask = np.zeros((9, 9, 4), dtype=np.bool_)
        
        # Only process if it's this player's turn
        if game.current_player != self.player:
            return action_mask
            
        for row in range(9):
            for col in range(9):
                piece = game.board[row][col]
                
                # Check if the piece belongs to current player
                is_player_piece = False
                if self.player == Player.WHITE and (piece == Piece.WHITE or piece == Piece.KING):
                    is_player_piece = True
                elif self.player == Player.BLACK and piece == Piece.BLACK:
                    is_player_piece = True
                
                if is_player_piece:
                    valid_moves = game.get_valid_moves(row, col)
                    
                    for move_row, move_col in valid_moves:
                        # Determine direction
                        if move_row < row:  # up
                            action_mask[row, col, 0] = True
                        elif move_col > col:  # right
                            action_mask[row, col, 1] = True
                        elif move_row > row:  # down
                            action_mask[row, col, 2] = True
                        elif move_col < col:  # left
                            action_mask[row, col, 3] = True
        
        return action_mask
    
    def select_action(self, game: TablutGame, epsilon=None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Select an action using epsilon-greedy policy"""
        if epsilon is None:
            epsilon = self.epsilon
        
        # Get state representation
        state = self.board_to_state(game)
        
        # Get valid actions mask
        valid_actions_mask = self.get_valid_actions_mask(game)
        
        # Check if there are any valid actions
        if not np.any(valid_actions_mask):
            return None
        
        # With probability epsilon, select a random valid action
        if random.random() < epsilon:
            # Find all valid actions
            valid_actions = []
            for row in range(9):
                for col in range(9):
                    for direction in range(4):
                        if valid_actions_mask[row, col, direction]:
                            # Convert direction to target position
                            if direction == 0:  # up
                                valid_actions.append(((row, col), (row-1, col)))
                            elif direction == 1:  # right
                                valid_actions.append(((row, col), (row, col+1)))
                            elif direction == 2:  # down
                                valid_actions.append(((row, col), (row+1, col)))
                            elif direction == 3:  # left
                                valid_actions.append(((row, col), (row, col-1)))
            
            if valid_actions:
                return random.choice(valid_actions)
            return None  # No valid actions
        
        # Otherwise, select the action with highest Q-value
        with torch.no_grad():
            q_values = self.q_network(state).squeeze(0)
        
        # Apply mask to filter invalid actions
        q_values_np = q_values.cpu().numpy()
        # Transpose from (4,9,9) to (9,9,4) to match mask dimensions
        q_values_np = np.transpose(q_values_np, (1, 2, 0))
        q_values_masked = np.where(valid_actions_mask, q_values_np, -np.inf)
        
        # Find the action with highest Q-value more efficiently
        flat_idx = np.argmax(q_values_masked.flatten())
        indices = np.unravel_index(flat_idx, q_values_masked.shape)
        row, col, direction = indices
        
        # Convert direction to target position
        if direction == 0:  # up
            best_action = ((row, col), (row-1, col))
        elif direction == 1:  # right
            best_action = ((row, col), (row, col+1))
        elif direction == 2:  # down
            best_action = ((row, col), (row+1, col))
        elif direction == 3:  # left
            best_action = ((row, col), (row, col-1))
        
        return best_action
    
    def calculate_reward(self, game: TablutGame, prev_board, action) -> float:
        """Calculate reward for the transition using the reward dictionary"""
        reward = self.rewards['move']  # Small penalty for each move
        
        # Check if game is over
        if game.is_game_over():
            winner = game.get_winner()
            if winner is None:  # Draw
                return self.rewards['draw']
            elif winner == self.player:  # Win
                return self.rewards['win']
            else:  # Lose
                return self.rewards['lose']
        
        # Check for captures or lost pieces
        prev_pieces = self.count_pieces(prev_board)
        curr_pieces = self.count_pieces(game.board)
        
        if self.player == Player.WHITE:
            # Pieces captured
            black_diff = prev_pieces['black'] - curr_pieces['black']
            if black_diff > 0:
                reward += black_diff * self.rewards['capture_piece']
            
            # Pieces lost
            white_diff = prev_pieces['white'] - curr_pieces['white']
            king_captured = prev_pieces['king'] > curr_pieces['king']
            
            if white_diff > 0:
                # Use subtraction for clarity with negative rewards
                reward -= abs(white_diff * self.rewards['lose_piece'])
            
            if king_captured:
                reward += self.rewards['lose']  # King captured is a loss
            
            # Check if king is approaching escape tiles
            if action and not king_captured:
                king_pos = self.find_king_position(game.board)
                
                if king_pos and action[0] == king_pos:  # If we moved the king
                    # Check if we're closer to any escape tile
                    min_dist_before = float('inf')
                    for escape_row, escape_col in game.ESCAPE_TILES:
                        dist = abs(action[1][0] - escape_row) + abs(action[1][1] - escape_col)
                        min_dist_before = min(min_dist_before, dist)
                    
                    min_dist_after = float('inf')
                    for escape_row, escape_col in game.ESCAPE_TILES:
                        dist = abs(king_pos[0] - escape_row) + abs(king_pos[1] - escape_col)
                        min_dist_after = min(min_dist_after, dist)
                    
                    if min_dist_after < min_dist_before:
                        reward += self.rewards['king_approach_escape']
        
        else:  # BLACK
            # Pieces captured
            white_diff = prev_pieces['white'] - curr_pieces['white']
            king_captured = prev_pieces['king'] > curr_pieces['king']
            
            if white_diff > 0:
                reward += white_diff * self.rewards['capture_piece']
            
            if king_captured:
                reward += self.rewards['king_capture']
            
            # Pieces lost
            black_diff = prev_pieces['black'] - curr_pieces['black']
            if black_diff > 0:
                # Use subtraction for clarity with negative rewards
                reward -= abs(black_diff * self.rewards['lose_piece'])
        
        return reward
    
    def count_pieces(self, board) -> Dict[str, int]:
        """Count pieces on the board"""
        counts = {'white': 0, 'black': 0, 'king': 0}
        
        for row in range(9):
            for col in range(9):
                piece = board[row][col]
                if piece == Piece.WHITE:
                    counts['white'] += 1
                elif piece == Piece.BLACK:
                    counts['black'] += 1
                elif piece == Piece.KING:
                    counts['king'] += 1
        
        return counts
    
    def update_model(self):
        """Update the model using a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample a batch from replay buffer
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return
            
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.cat(states, dim=0)
        next_states = torch.cat(next_states, dim=0)
        rewards = torch.FloatTensor(rewards).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Convert actions to indices for tensor access
        action_indices = []
        for action in actions:
            if action is None:
                action_indices.append((0, 0, 0))  # Default value for None actions
                continue
                
            from_pos, to_pos = action
            row, col = from_pos
            
            # Determine direction
            direction = -1
            if to_pos[0] < row:  # up
                direction = 0
            elif to_pos[1] > col:  # right
                direction = 1
            elif to_pos[0] > row:  # down
                direction = 2
            elif to_pos[1] < col:  # left
                direction = 3
            
            if direction == -1:
                action_indices.append((0, 0, 0))  # Default for invalid direction
            else:
                action_indices.append((direction, row, col))
        
        # Get current Q values - Note the order: output is [batch, channels, height, width]
        current_q = self.q_network(states)
        current_q_values = []
        
        for i, (direction, row, col) in enumerate(action_indices):
            if direction == -1:
                # Use a default value for None actions
                current_q_values.append(torch.zeros(1, device=self.device))
            else:
                current_q_values.append(current_q[i, direction, row, col].unsqueeze(0))
        
        current_q_values = torch.cat(current_q_values)
        
        # Get next state values using target network and compute target Q values
        with torch.no_grad():
            # Compute max Q value for each state over all actions
            next_q = self.target_network(next_states)
            # Get max across action dimension (dim=1), then spatial dimensions
            max_next_q, _ = next_q.reshape(next_q.size(0), next_q.size(1), -1).max(2)
            max_next_q, _ = max_next_q.max(1)
            # Compute target Q values
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Calculate loss and update
        loss = self.loss_fn(current_q_values, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        # Add gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.training_step += 1
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return loss.item()
    
    def make_move(self, game: TablutGame, epsilon=None):
        """Make a move in the game
        
        Args:
            game: The game state
            epsilon: Optional override for exploration rate, if None uses self.eval_epsilon
        """
        # Check if it's our turn
        if game.current_player != self.player:
            return f"Not {self.player.value}'s turn"
            
        # Save the current board state for reward calculation
        prev_board = copy.deepcopy(game.board)
        
        # Select action
        if epsilon is None:
            epsilon = self.eval_epsilon
            
        action = self.select_action(game, epsilon=epsilon)
        
        if action is None:
            return "No valid moves available"
        
        from_pos, to_pos = action
        success, error = game.move_piece(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
        
        if not success:
            return f"Failed to make move: {error}"
        
        return f"{self.player.value} moved from {from_pos} to {to_pos}"
    
    def train(self, num_episodes=1000, opponent=None):
        """Train the agent for a specified number of episodes"""
        for episode in range(num_episodes):
            game = TablutGame()
            
            # Set move callbacks
            if self.player == Player.WHITE:
                game.set_move_callback(self.training_move, Player.WHITE)
                if opponent:
                    game.set_move_callback(opponent.make_move, Player.BLACK)
            else:
                if opponent:
                    game.set_move_callback(opponent.make_move, Player.WHITE)
                game.set_move_callback(self.training_move, Player.BLACK)
            
            # Play a full game
            state = self.board_to_state(game)
            done = False
            
            while not done:
                if game.current_player == self.player:
                    # Agent's turn
                    action = self.select_action(game)
                    
                    if action is None:
                        # No valid moves
                        break
                    
                    from_pos, to_pos = action
                    prev_board = copy.deepcopy(game.board)
                    
                    # Make move
                    success, _ = game.move_piece(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
                    
                    if not success:
                        continue
                    
                    # Get next state
                    next_state = self.board_to_state(game)
                    
                    # Check if game is over
                    done = game.is_game_over()
                    
                    # Calculate reward
                    reward = self.calculate_reward(game, prev_board, action)
                    
                    # Add to replay buffer
                    self.replay_buffer.add(state, action, reward, next_state, done)
                    
                    # Update state
                    state = next_state
                    
                    # Update model
                    self.update_model()
                else:
                    # Opponent's turn
                    game.notify_move_needed()
                    
                    # Get next state after opponent's move
                    next_state = self.board_to_state(game)
                    
                    # Check if game is over
                    done = game.is_game_over()
                    
                    # Update state
                    state = next_state
            
            # Update stats
            self.episodes_played += 1
            
            if game.is_game_over():
                winner = game.get_winner()
                if winner is None:
                    self.draws += 1
                elif winner == self.player:
                    self.wins += 1
                else:
                    self.losses += 1
            
            # Log progress
            if (episode + 1) % 10 == 0:
                win_rate = self.wins / self.episodes_played
                print(f"Episode {episode+1}/{num_episodes}, Win rate: {win_rate:.3f}, "
                      f"Epsilon: {self.epsilon:.3f}")
    
    def training_move(self, game: TablutGame):
        """Make a move during training (used for callbacks)"""
        # This is just a wrapper around make_move
        return self.make_move(game, epsilon=self.epsilon)
    
    def save_model(self, path):
        """Save the model to a file"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'eval_epsilon': self.eval_epsilon,
            'training_step': self.training_step,
            'episodes_played': self.episodes_played,
            'wins': self.wins,
            'losses': self.losses,
            'draws': self.draws,
            'rewards': self.rewards  # Save reward values
        }, path)
    
    def load_model(self, path):
        """Load the model from a file"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        if 'eval_epsilon' in checkpoint:
            self.eval_epsilon = checkpoint['eval_epsilon']
        self.training_step = checkpoint['training_step']
        self.episodes_played = checkpoint['episodes_played']
        self.wins = checkpoint['wins']
        self.losses = checkpoint['losses']
        self.draws = checkpoint['draws']
        # Load rewards if available in the checkpoint
        if 'rewards' in checkpoint:
            self.rewards = checkpoint['rewards']

class RandomAgent:
    """A simple agent that makes random moves for testing and training"""
    def __init__(self, player: Player):
        self.player = player
    
    def make_move(self, game: TablutGame):
        """Make a random valid move"""
        valid_moves = []
        
        for row in range(9):
            for col in range(9):
                piece = game.board[row][col]
                
                # Check if piece belongs to this player
                is_player_piece = False
                if self.player == Player.WHITE and (piece == Piece.WHITE or piece == Piece.KING):
                    is_player_piece = True
                elif self.player == Player.BLACK and piece == Piece.BLACK:
                    is_player_piece = True
                
                if is_player_piece:
                    moves = game.get_valid_moves(row, col)
                    for move in moves:
                        valid_moves.append(((row, col), move))
        
        if not valid_moves:
            return "No valid moves available"
        
        # Select a random move
        from_pos, to_pos = random.choice(valid_moves)
        success, error = game.move_piece(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
        
        if not success:
            return f"Failed to make move: {error}"
        
        return f"{self.player.value} moved from {from_pos} to {to_pos}"

def process_player_turn(game, agent, agent_state, prev_action, prev_board, opponent_agent=None, opponent_state=None):
    """Process a player's turn and handle the updates for both players
    
    Args:
        game: The current game state
        agent: The agent whose turn it is
        agent_state: The current state for the agent
        prev_action: The previous action of the agent (or None)
        prev_board: The previous board state for the agent (or None)
        opponent_agent: The opponent agent
        opponent_state: The current state for the opponent
        
    Returns:
        Tuple of (new_state, action, board, done)
    """
    # Select action
    action = agent.select_action(game)
    
    if action is None:
        return None, None, None, True
    
    # Save board before move
    board_copy = copy.deepcopy(game.board)
    
    # Make move
    from_pos, to_pos = action
    success, _ = game.move_piece(from_pos[0], from_pos[1], to_pos[0], to_pos[1])
    
    if not success:
        return agent_state, prev_action, prev_board, False
    
    # Get new state
    new_state = agent.board_to_state(game)
    done = game.is_game_over()
    
    # Calculate reward and add to replay buffer only if there was a previous action
    if prev_action is not None and prev_board is not None:
        reward = agent.calculate_reward(game, prev_board, prev_action)
        agent.replay_buffer.add(agent_state, prev_action, reward, new_state, done)
        agent.update_model()
    
    # Update opponent state if provided
    if opponent_agent and opponent_state:
        new_opponent_state = opponent_agent.board_to_state(game)
        
        # Return updated states and action info
        return new_state, action, board_copy, done, new_opponent_state
    
    return new_state, action, board_copy, done

def train_agents(white_agent, black_agent, num_episodes=1000, update_freq=5):
    """Train both agents by playing against each other
    
    Args:
        white_agent: The white player agent
        black_agent: The black player agent
        num_episodes: Number of episodes to train for
        update_freq: How often to update the models (every N moves)
    """
    for episode in range(num_episodes):
        game = TablutGame()
        
        # Initialize states
        white_state = white_agent.board_to_state(game)
        black_state = black_agent.board_to_state(game)
        
        # Track last actions and board states for reward calculation
        white_action = None
        white_prev_board = None
        black_action = None
        black_prev_board = None
        
        # Track transitions for batch updates
        white_transitions = []
        black_transitions = []
        
        # Play until game over
        move_count = 0
        while not game.is_game_over():
            move_count += 1
            
            if game.current_player == Player.WHITE:
                # White's turn
                result = process_player_turn(
                    game, white_agent, white_state, white_action, white_prev_board
                )
                
                if result is None:  # No valid moves
                    break
                    
                white_state, white_action, white_prev_board, done = result
                
                if done:
                    break
                
                # Update black's state after white's move
                black_state = black_agent.board_to_state(game)
                
                # Process black's previous action if it exists
                if black_action is not None and black_prev_board is not None:
                    black_reward = black_agent.calculate_reward(game, black_prev_board, black_action)
                    black_transitions.append((black_state, black_action, black_reward, black_state, done))
                    
                    # Reset black's tracking variables
                    black_action = None
                    black_prev_board = None
            else:
                # Black's turn
                result = process_player_turn(
                    game, black_agent, black_state, black_action, black_prev_board
                )
                
                if result is None:  # No valid moves
                    break
                    
                black_state, black_action, black_prev_board, done = result
                
                if done:
                    break
                
                # Update white's state after black's move
                white_state = white_agent.board_to_state(game)
                
                # Process white's previous action if it exists
                if white_action is not None and white_prev_board is not None:
                    white_reward = white_agent.calculate_reward(game, white_prev_board, white_action)
                    white_transitions.append((white_state, white_action, white_reward, white_state, done))
                    
                    # Reset white's tracking variables
                    white_action = None
                    white_prev_board = None
            
            # Periodically update models during gameplay
            if move_count % update_freq == 0:
                # Add all transitions to replay buffers
                for state, action, reward, next_state, done in white_transitions:
                    white_agent.replay_buffer.add(state, action, reward, next_state, done)
                for state, action, reward, next_state, done in black_transitions:
                    black_agent.replay_buffer.add(state, action, reward, next_state, done)
                
                # Update models
                white_agent.update_model()
                black_agent.update_model()
                
                # Clear transitions
                white_transitions = []
                black_transitions = []
        
        # Final updates with any remaining transitions
        for state, action, reward, next_state, done in white_transitions:
            white_agent.replay_buffer.add(state, action, reward, next_state, done)
        for state, action, reward, next_state, done in black_transitions:
            black_agent.replay_buffer.add(state, action, reward, next_state, done)
        
        white_agent.update_model()
        black_agent.update_model()
        
        # Update stats for both agents
        white_agent.episodes_played += 1
        black_agent.episodes_played += 1
        
        # Record game result
        if game.is_game_over():
            winner = game.get_winner()
            if winner is None:
                white_agent.draws += 1
                black_agent.draws += 1
            elif winner == Player.WHITE:
                white_agent.wins += 1
                black_agent.losses += 1
            else:
                white_agent.losses += 1
                black_agent.wins += 1
        
        # Log progress
        if (episode + 1) % 10 == 0:
            white_win_rate = white_agent.wins / white_agent.episodes_played
            black_win_rate = black_agent.wins / black_agent.episodes_played
            print(f"Episode {episode+1}/{num_episodes}, "
                  f"White win rate: {white_win_rate:.3f}, "
                  f"Black win rate: {black_win_rate:.3f}, "
                  f"White epsilon: {white_agent.epsilon:.3f}, "
                  f"Black epsilon: {black_agent.epsilon:.3f}")

# Example of customizing rewards
def create_aggressive_black_agent():
    """Create a black agent with more aggressive reward settings"""
    custom_rewards = {
        'capture_piece': 1.5,     # Increased reward for capturing pieces
        'king_capture': 4.0,      # Much higher reward for capturing king
        'lose_piece': -0.5,       # Less penalty for losing pieces (encouraging risk)
    }
    return TablutRLAgent(Player.BLACK, rewards=custom_rewards)

def create_defensive_white_agent():
    """Create a white agent with more defensive reward settings"""
    custom_rewards = {
        'king_approach_escape': 0.5,  # Higher reward for king approaching escape
        'lose_piece': -1.5,           # Higher penalty for losing pieces
    }
    return TablutRLAgent(Player.WHITE, rewards=custom_rewards)

if __name__ == "__main__":
    # Create agents
    white_agent = TablutRLAgent(Player.WHITE)
    black_agent = TablutRLAgent(Player.BLACK)
    
    # Alternatively, create agents with custom reward structures
    # white_agent = create_defensive_white_agent()
    # black_agent = create_aggressive_black_agent()
    
    # Train agents against each other
    print("Training agents against each other...")
    train_agents(white_agent, black_agent, num_episodes=500)
    
    # Save trained models
    white_agent.save_model("white_agent.pth")
    black_agent.save_model("black_agent.pth")
    
    # Create a game with RL agents
    game = TablutGame()
    game.set_move_callback(white_agent.make_move, Player.WHITE)
    game.set_move_callback(black_agent.make_move, Player.BLACK)
    
    # Run the game
    visualizer = GameVisualizer()
    visualizer.run(game, white_player_type=PlayerType.AI, black_player_type=PlayerType.AI) 