import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from tablut import TablutGame
from utils import Player, Piece
import copy
from typing import List, Dict, Tuple
from utils import GameVisualizer, PlayerType
import torch.nn.functional as F
import time


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # Input: 9x9 board state (one-hot encoded for each piece type)
        # 6 channels: empty, white, black, king, castle/camp, escape
        self.conv1 = nn.Conv2d(6, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # Two heads: policy (action selection) and value (state evaluation)
        self.policy_head = nn.Sequential(
            nn.Linear(128 * 9 * 9, 1024),
            nn.ReLU(),
            nn.Linear(1024, 648)  # 81 possible from positions * 8 directions
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(128 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(-1, 128 * 9 * 9)
        
        policy = self.policy_head(x)
        value = self.value_head(x)
        
        return policy, value

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class RLAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = ReplayBuffer()
        
        self.batch_size = 32
        self.gamma = 0.99
        self.eps_start = 0.9
        self.eps_end = 0.05
        self.eps_decay = 200
        self.target_update = 10
        self.steps_done = 0
        
    def board_to_tensor(self, game):
        # Convert board to 6-channel tensor
        state = np.zeros((6, 9, 9))
        
        for i in range(9):
            for j in range(9):
                piece = game.board[i][j]
                if piece == Piece.EMPTY:
                    state[0][i][j] = 1
                elif piece == Piece.WHITE:
                    state[1][i][j] = 1
                elif piece == Piece.BLACK:
                    state[2][i][j] = 1
                elif piece == Piece.KING:
                    state[3][i][j] = 1
                elif piece in [Piece.CASTLE, Piece.CAMP]:
                    state[4][i][j] = 1
                elif piece == Piece.ESCAPE:
                    state[5][i][j] = 1
                    
        return torch.FloatTensor(state).unsqueeze(0).to(self.device)
    
    def select_action(self, game):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            np.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                state = self.board_to_tensor(game)
                policy, _ = self.policy_net(state)
                moves = self.get_possible_moves(game, Player.BLACK)
                if not moves:
                    return None
                    
                # Filter valid moves
                valid_actions = []
                for move in moves:
                    from_pos = move['from']
                    to_pos = move['to']
                    action_idx = from_pos[0] * 9 * 8 + from_pos[1] * 8 + self.direction_to_idx(from_pos, to_pos)
                    valid_actions.append(action_idx)
                
                # Select highest-valued valid action
                policy = policy.squeeze()
                valid_policy = policy[valid_actions]
                best_idx = valid_actions[valid_policy.argmax()]
                
                # Convert back to move
                from_row = best_idx // (9 * 8)
                from_col = (best_idx % (9 * 8)) // 8
                direction = self.idx_to_direction(best_idx % 8)
                to_row = from_row + direction[0]
                to_col = from_col + direction[1]
                
                return {'from': (from_row, from_col), 'to': (to_row, to_col)}
        else:
            # Random valid move
            moves = self.get_possible_moves(game, Player.BLACK)
            if not moves:
                return None
            return random.choice(moves)
    
    def direction_to_idx(self, from_pos, to_pos):
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        return directions.index((dr,dc))
        
    def idx_to_direction(self, idx):
        directions = [(0,1), (0,-1), (1,0), (-1,0)]
        return directions[idx]
    
    def get_possible_moves(self, game, player):
        moves = []
        for row in range(9):
            for col in range(9):
                piece = game.board[row][col]
                if piece == Piece.BLACK:
                    possible_destinations = game.get_valid_moves(row, col)
                    for dest_row, dest_col in possible_destinations:
                        moves.append({
                            'from': (row, col),
                            'to': (dest_row, dest_col)
                        })
        return moves
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
            
        transitions = self.memory.sample(self.batch_size)
        batch = list(zip(*transitions))
        
        state_batch = torch.cat(batch[0])
        action_batch = torch.tensor(batch[1], device=self.device)
        reward_batch = torch.tensor(batch[2], device=self.device)
        next_state_batch = torch.cat(batch[3])
        done_batch = torch.tensor(batch[4], device=self.device)
        
        policy, state_values = self.policy_net(state_batch)
        next_policy, next_values = self.target_net(next_state_batch)
        
        # Compute Q values
        state_action_values = policy.gather(1, action_batch)
        next_state_values = next_values.max(1)[0].detach()
        expected_state_action_values = (next_state_values * self.gamma * (1 - done_batch)) + reward_batch
        
        # Compute loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def calculate_reward(game: TablutGame, action: dict, player: Player) -> float:
        """
        Calculates the reward for a given move, incorporating multiple strategic elements.
        """
        reward = 0.0
        # Reward coefficients
        WIN_REWARD = 100
        LOSS_PENALTY = -100
        MATERIAL_WEIGHT = 5
        KEY_SQUARE_WEIGHT = 3
        KING_NEIGHBOR_WEIGHT = 8
        KING_ESCAPE_PENALTY = 10
        KING_PROTECTOR_WEIGHT = 8
        KING_ESCAPE_PROGRESS_WEIGHT = 5
        CONNECTED_PIECE_WEIGHT = 2
        BLACK_ESCAPE_CONTROL_WEIGHT = 5
        WHITE_ESCAPE_CLEAR_WEIGHT = 3
        MOBILITY_WEIGHT = 0.5
        CENTER_CONTROL_WEIGHT = 3
        PIECE_PROTECTION_WEIGHT = 2

        reward = 0.0
        
        # 1. Basic game outcome rewards
        if game.is_game_over():
            winner = game.get_winner()
            if winner == player:
                reward += WIN_REWARD  # Win
            else:
                reward += LOSS_PENALTY  # Loss
            return reward

        # 2. Piece count and material advantage
        black_count = sum(row.count(Piece.BLACK) for row in game.board)
        white_count = sum(row.count(Piece.WHITE) for row in game.board)
        if player == Player.BLACK:
            reward += (black_count - white_count) * MATERIAL_WEIGHT
        else:
            reward += (white_count - black_count) * MATERIAL_WEIGHT

        # 3. Control of key squares
        key_squares = [(3,3), (3,4), (3,5), (4,3), (4,5), (5,3), (5,4), (5,5)]
        for row, col in key_squares:
            piece = game.board[row][col]
            if piece == Piece.BLACK and player == Player.BLACK:
                reward += KEY_SQUARE_WEIGHT
            elif piece == Piece.WHITE and player == Player.WHITE:
                reward += KEY_SQUARE_WEIGHT

        # 4. Find king position
        king_pos = None
        for row in range(9):
            for col in range(9):
                if game.board[row][col] == Piece.KING:
                    king_pos = (row, col)
                    break
            if king_pos:
                break

        if king_pos:
            king_row, king_col = king_pos
            
            # 5. King-specific rewards
            if player == Player.BLACK:
                # Reward for surrounding king
                black_neighbors = 0
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    new_row, new_col = king_row + dr, king_col + dc
                    if 0 <= new_row < 9 and 0 <= new_col < 9:
                        if game.board[new_row][new_col] == Piece.BLACK:
                            black_neighbors += 1
                reward += black_neighbors * KING_NEIGHBOR_WEIGHT

                # Penalize king being near escape
                for escape_row, escape_col in game.ESCAPE_TILES:
                    distance = abs(king_row - escape_row) + abs(king_col - escape_col)
                    if distance <= 2:
                        reward -= (3 - distance) * KING_ESCAPE_PENALTY
            
            else:  # WHITE player
                # Reward for protecting king
                white_protectors = 0
                for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                    new_row, new_col = king_row + dr, king_col + dc
                    if 0 <= new_row < 9 and 0 <= new_col < 9:
                        if game.board[new_row][new_col] == Piece.WHITE:
                            white_protectors += 1
                reward += white_protectors * KING_PROTECTOR_WEIGHT

                # Reward for king moving towards escape
                min_escape_distance = float('inf')
                for escape_row, escape_col in game.ESCAPE_TILES:
                    distance = abs(king_row - escape_row) + abs(king_col - escape_col)
                    min_escape_distance = min(min_escape_distance, distance)
                reward += (8 - min_escape_distance) * KING_ESCAPE_PROGRESS_WEIGHT

        # 6. Piece formation rewards
        for row in range(9):
            for col in range(9):
                if ((game.board[row][col] == Piece.BLACK and player == Player.BLACK) or
                    (game.board[row][col] == Piece.WHITE and player == Player.WHITE)):
                    # Reward for connected pieces
                    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < 9 and 0 <= new_col < 9:
                            if game.board[new_row][new_col] == game.board[row][col]:
                                reward += CONNECTED_PIECE_WEIGHT

        # 7. Control of escape routes
        if player == Player.BLACK:
            for row, col in game.ESCAPE_TILES:
                if game.board[row][col] == Piece.BLACK:
                    reward += BLACK_ESCAPE_CONTROL_WEIGHT
        else:
            escape_routes_clear = 0
            for row, col in game.ESCAPE_TILES:
                if game.board[row][col] == Piece.EMPTY:
                    escape_routes_clear += 1
            reward += escape_routes_clear * WHITE_ESCAPE_CLEAR_WEIGHT

        # 8. Mobility reward
        valid_moves = len(game.get_valid_moves_for_player(player))
        reward += valid_moves * MOBILITY_WEIGHT

        # 9. Center control for black
        if player == Player.BLACK:
            center_control = 0
            for row in range(3, 6):
                for col in range(3, 6):
                    if game.board[row][col] == Piece.BLACK:
                        center_control += 1
            reward += center_control * CENTER_CONTROL_WEIGHT

        # 10. Piece safety
        for row in range(9):
            for col in range(9):
                current_piece = game.board[row][col]
                if ((current_piece == Piece.BLACK and player == Player.BLACK) or
                    (current_piece == Piece.WHITE and player == Player.WHITE)):
                    # Check if piece is protected
                    is_protected = False
                    for dr, dc in [(0,1), (1,0), (0,-1), (-1,0)]:
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < 9 and 0 <= new_col < 9:
                            if game.board[new_row][new_col] == current_piece:
                                is_protected = True
                                break
                    if is_protected:
                        reward += PIECE_PROTECTION_WEIGHT

        return reward

    def train(self, num_episodes=1000):
        for episode in range(num_episodes):
            game = TablutGame()
            total_reward = 0
            done = False
            
            while not done:
                state = self.board_to_tensor(game)
                action = self.select_action(game)
                
                if not action:
                    break
                    
                # Make move
                success, _ = game.move_piece(action['from'][0], action['from'][1], 
                                        action['to'][0], action['to'][1])
                
                if not success:
                    reward = -10  # Invalid move penalty
                    done = True
                else:
                    # Calculate reward using our new function
                    reward = calculate_reward(game, action, game.current_player)
                    done = game.is_game_over()
                
                next_state = self.board_to_tensor(game)
                
                # Store transition
                self.memory.push(state, action, reward, next_state, done)
                
                total_reward += reward
                
                # Optimize model
                self.optimize_model()
                
                if done:
                    break
                    
                # Let opponent (white) make a move
                game.notify_move_needed()
            
            # Update target network
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                
            print(f"Episode {episode}: Total reward = {total_reward}")
            
    def make_move(self, game, player):
        """Interface method for game integration"""
        action = self.select_action(game)
        if not action:
            return "No valid moves available"
            
        success, log = game.move_piece(action['from'][0], action['from'][1],
                                     action['to'][0], action['to'][1])
        return log

def make_rl_move(game):
    """Callback function for the RL agent"""
    agent = RLAgent()
    return agent.make_move(game, game.current_player)

if __name__ == "__main__":
    # Train the agent
    agent = RLAgent()
    agent.train(num_episodes=100)
    
    # Play a game with trained agent
    game = TablutGame()
    visualizer = GameVisualizer()
    game.set_move_callback(make_rl_move, Player.BLACK)
    game.set_move_callback(make_rl_move, Player.BLACK)
    visualizer.run(game, white_player_type=PlayerType.GUI, black_player_type=PlayerType.HEURISTIC)
