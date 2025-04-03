import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import os
from tablut import TablutGame
from utils import Player, Piece, PlayerType

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
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Output layer: For each of the 9x9 positions, predict Q-values for 4 directions
        self.output = nn.Conv2d(128, 4, kernel_size=1)
        
        # Activation functions
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # CNN layers
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        
        # Output Q-values for each position and direction
        q_values = self.output(x)
        
        return q_values

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
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
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.0001)
        
        # Initialize replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training parameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.update_target_every = 10  # Update target network every N episodes
        self.episodes_count = 0
        
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
        """Convert action tuple to direction index"""
        from_row, from_col, to_row, to_col = action
        
        # Determine direction (0: up, 1: right, 2: down, 3: left)
        if from_row > to_row:  # Moving up
            direction = 0
        elif from_col < to_col:  # Moving right
            direction = 1
        elif from_row < to_row:  # Moving down
            direction = 2
        else:  # Moving left
            direction = 3
        
        return from_row, from_col, direction

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
            q_values = self.q_network(state).squeeze(0)
        
        # Find best action among valid actions
        best_q_value = float('-inf')
        best_action = None
        
        for action in valid_actions:
            from_row, from_col, direction = self.action_to_index(action)
            q_value = q_values[direction, from_row, from_col].item()
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        return best_action

    def train(self):
        """Train the Q-network using a batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert actions to indices for easy access
        action_indices = []
        for action in actions:
            from_row, from_col, direction = self.action_to_index(action)
            action_indices.append((direction, from_row, from_col))
        
        # Compute current Q-values
        curr_q_values = self.q_network(states)
        curr_q = torch.zeros(self.batch_size, device=self.device)
        
        for i in range(self.batch_size):
            direction, from_row, from_col = action_indices[i]
            curr_q[i] = curr_q_values[i, direction, from_row, from_col]
        
        # Compute target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q = torch.max(next_q_values.view(self.batch_size, -1), dim=1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
        
        # Compute loss and update network
        loss = nn.MSELoss()(curr_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def update_target_network(self):
        """Update the target network with current Q-network weights"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_model(self, filename):
        """Save model weights to file"""
        os.makedirs('models', exist_ok=True)
        torch.save(self.q_network.state_dict(), f'models/{filename}')

    def load_model(self, filename):
        """Load model weights from file"""
        self.q_network.load_state_dict(torch.load(f'models/{filename}'))
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

def train_agent(white_agent=None, black_agent=None, num_episodes=1000):
    """Train agent(s) by playing games"""
    # Initialize agents if not provided
    if white_agent is None:
        white_agent = TablutRLAgent(Player.WHITE)
    if black_agent is None:
        black_agent = TablutRLAgent(Player.BLACK)
    
    # Training stats
    white_wins = 0
    black_wins = 0
    draws = 0
    
    for episode in range(num_episodes):
        game = TablutGame()
        
        # Set move callbacks
        white_callback = lambda g: rl_agent_move(g, white_agent)
        black_callback = lambda g: rl_agent_move(g, black_agent)
        
        game.set_move_callback(white_callback, Player.WHITE)
        game.set_move_callback(black_callback, Player.BLACK)
        
        # Reset state
        current_state = board_to_state_tensor(game.board)
        done = False
        total_reward = {Player.WHITE: 0, Player.BLACK: 0}
        transitions = {Player.WHITE: [], Player.BLACK: []}
        
        # Game loop
        while not done:
            current_player = game.current_player
            current_agent = white_agent if current_player == Player.WHITE else black_agent
            
            # Store the board state before the move
            pre_move_board = [row[:] for row in game.board]
            pre_pieces = {
                Player.WHITE: sum(1 for row in pre_move_board for piece in row 
                                if piece in [Piece.WHITE, Piece.KING]),
                Player.BLACK: sum(1 for row in pre_move_board for piece in row 
                                if piece == Piece.BLACK)
            }
            
            # Select and perform action
            action = current_agent.select_action(game, training=True)
            
            if action:
                from_row, from_col, to_row, to_col = action
                game.move_piece(from_row, from_col, to_row, to_col)
            else:
                # No valid moves - opponent wins
                done = True
                if current_player == Player.WHITE:
                    black_wins += 1
                else:
                    white_wins += 1
                continue
            
            # Get new state
            next_state = board_to_state_tensor(game.board)
            
            # Calculate reward
            reward = -0.1  # Move penalty
            
            # Check for captures
            post_pieces = {
                Player.WHITE: sum(1 for row in game.board for piece in row 
                                if piece in [Piece.WHITE, Piece.KING]),
                Player.BLACK: sum(1 for row in game.board for piece in row 
                                if piece == Piece.BLACK)
            }
            
            # Reward for capturing opponent pieces
            pieces_captured = pre_pieces[Player.BLACK if current_player == Player.WHITE else Player.WHITE] - \
                             post_pieces[Player.BLACK if current_player == Player.WHITE else Player.WHITE]
            reward += pieces_captured * 1.0
            
            # Penalty for losing pieces
            pieces_lost = pre_pieces[current_player] - post_pieces[current_player]
            reward -= pieces_lost * 1.0
            
            # Check if game is over
            if game.is_game_over():
                done = True
                winner = game.get_winner()
                
                if winner is not None:
                    if winner == current_player:
                        reward += 3.0  # Win reward
                        if winner == Player.WHITE:
                            white_wins += 1
                        else:
                            black_wins += 1
                    else:
                        reward -= 3.0  # Lose penalty
                        if winner == Player.WHITE:
                            white_wins += 1
                        else:
                            black_wins += 1
                # Draw
                else:
                    draws += 1
            
            # Store transition
            transitions[current_player].append((current_state, action, reward, next_state, done))
            
            # Update current state
            current_state = next_state
            
            # Accumulate reward
            total_reward[current_player] += reward
        
        # Add all transitions to replay buffer
        for player, player_transitions in transitions.items():
            agent = white_agent if player == Player.WHITE else black_agent
            for state, action, reward, next_state, done in player_transitions:
                agent.replay_buffer.add(state, action, reward, next_state, done)
            
            # Train the agent
            loss = agent.train()
            
            # Decay epsilon
            agent.decay_epsilon()
        
        # Update target networks periodically
        white_agent.episodes_count += 1
        black_agent.episodes_count += 1
        
        if white_agent.episodes_count % white_agent.update_target_every == 0:
            white_agent.update_target_network()
        
        if black_agent.episodes_count % black_agent.update_target_every == 0:
            black_agent.update_target_network()
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode+1}/{num_episodes}")
            print(f"White Wins: {white_wins}, Black Wins: {black_wins}, Draws: {draws}")
            print(f"White Epsilon: {white_agent.epsilon:.4f}, Black Epsilon: {black_agent.epsilon:.4f}")
            print(f"White Total Reward: {total_reward[Player.WHITE]:.2f}, Black Total Reward: {total_reward[Player.BLACK]:.2f}")
            print("-" * 40)
        
        # Save models periodically
        if (episode + 1) % 100 == 0:
            white_agent.save_model(f"white_agent_ep{episode+1}.pt")
            black_agent.save_model(f"black_agent_ep{episode+1}.pt")
    
    # Save final models
    white_agent.save_model("white_agent_final.pt")
    black_agent.save_model("black_agent_final.pt")
    
    return white_agent, black_agent

def visualize_game(white_agent, black_agent):
    """Run a visual game between two agents"""
    from utils import GameVisualizer
    
    # Create a new game
    game = TablutGame()
    
    # Set up the game visualizer
    visualizer = GameVisualizer()
    
    # Set move callbacks for the agents
    white_callback = lambda g: rl_agent_move(g, white_agent)
    black_callback = lambda g: rl_agent_move(g, black_agent)
    
    game.set_move_callback(white_callback, Player.WHITE)
    game.set_move_callback(black_callback, Player.BLACK)
    
    # Set the game to use RL agents instead of GUI input
    visualizer.run(game, 
                  white_player_type=PlayerType.RL, 
                  black_player_type=PlayerType.RL)

def main():
    # Initialize agents
    white_agent = TablutRLAgent(Player.WHITE)
    black_agent = TablutRLAgent(Player.BLACK)
    
    # Train agents
    print("Starting training...")
    white_agent, black_agent = train_agent(white_agent, black_agent, num_episodes=1000)
    print("Training completed!")
    
    # Evaluate the trained agent
    def evaluate_agent(agent, opponent_agent, num_games=100):
        wins = 0
        for _ in range(num_games):
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
            
            # Play until game over
            while not game.is_game_over():
                current_player = game.current_player
                if current_player == Player.WHITE:
                    white_callback(game)
                else:
                    black_callback(game)
            
            # Check winner
            winner = game.get_winner()
            if winner == agent.player:
                wins += 1
        
        return wins / num_games
    
    # Evaluate against random agent
    random_agent = TablutRLAgent(Player.BLACK if white_agent.player == Player.WHITE else Player.WHITE)
    random_agent.epsilon = 1.0  # Always take random actions
    
    white_win_rate = evaluate_agent(white_agent, random_agent)
    black_win_rate = evaluate_agent(black_agent, random_agent)
    
    print(f"White Agent Win Rate vs Random: {white_win_rate:.2f}")
    print(f"Black Agent Win Rate vs Random: {black_win_rate:.2f}")
    
    # Save final models
    white_agent.save_model("white_agent_final.pt")
    black_agent.save_model("black_agent_final.pt")

    # Run a visual game between the trained agents
    print("\nRunning a visual game between trained agents...")
    visualize_game(white_agent, black_agent)

if __name__ == "__main__":
    main()

