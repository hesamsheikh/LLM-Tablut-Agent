import os
import random
import numpy as np
import time
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Import existing code
from tablut import TablutGame, Player, Piece


###############################################################################
# 1) ENVIRONMENT WRAPPER (SELF-PLAY) WITH REWARD SHAPING
###############################################################################
class TablutEnv:
    """
    A self-play environment for Tablut, wrapping TablutGame.
    One agent controls whichever player is 'current_player'.
    
    Configurable rewards for various game events.
    """
    def __init__(self, move_limit=40, rewards=None):
        self.game = TablutGame()
        self.move_limit = move_limit
        # 9x9 => 81 squares for "from", 81 squares for "to" => 6561 total discrete actions
        self.num_actions = 81 * 81
        # Observations: [Black, White, King, Camp, Escape, Castle]
        self.obs_shape = (6, 9, 9)
        
        # Initialize rewards, use defaults if not specified
        self.rewards = rewards
            
        self.reset()

    def reset(self):
        self.game = TablutGame()
        self.steps_taken = 0
        # Add capture counter
        self.captures_count = 0
        return self._get_observation()

    def step(self, action):
        """
        Decode action => (fr, fc, tr, tc), apply it to TablutGame, 
        compute reward, and return (next_obs, reward, done, info).
        """
        from_index = action // 81
        to_index   = action % 81
        fr, fc = divmod(from_index, 9)
        tr, tc = divmod(to_index, 9)

        current_player = self.game.current_player
        reward = 0.0
        done = False
        info = {"invalid_move": False, "captured_pieces": 0, "end_reason": None}

        # ----------------------------
        # A) Count opponent pieces BEFORE move (for capture reward)
        # ----------------------------
        if current_player == Player.WHITE:
            pieces_before = sum(cell == Piece.BLACK for row in self.game.board for cell in row)
        else:  # current_player == Player.BLACK
            pieces_before = sum(cell in [Piece.WHITE, Piece.KING] for row in self.game.board for cell in row)

        # ----------------------------
        # B) If White, track King distance BEFORE
        # ----------------------------
        king_dist_before = None
        if current_player == Player.WHITE:
            king_dist_before = self._king_distance_to_closest_escape()

        # Attempt the move
        success, end_reason = self.game.move_piece(fr, fc, tr, tc)
        if not success:
            # Invalid move
            reward += self.rewards['INVALID_MOVE']
            info["invalid_move"] = True
        else:
            # ----------------------------
            # C) Capture Reward
            # ----------------------------
            # Compare piece counts after move
            if current_player == Player.WHITE:
                pieces_after = sum(cell == Piece.BLACK for row in self.game.board for cell in row)
                if pieces_after < pieces_before:
                    # at least one black piece was captured
                    pieces_captured = pieces_before - pieces_after
                    reward += self.rewards['CAPTURE_PIECE']
                    self.captures_count += pieces_captured
                    info["captured_pieces"] = pieces_captured
            else:
                pieces_after = sum(cell in [Piece.WHITE, Piece.KING] for row in self.game.board for cell in row)
                if pieces_after < pieces_before:
                    # captured a white soldier or the king
                    pieces_captured = pieces_before - pieces_after
                    reward += self.rewards['CAPTURE_PIECE']
                    self.captures_count += pieces_captured
                    info["captured_pieces"] = pieces_captured

            # ----------------------------
            # D) King Moves Closer
            # ----------------------------
            if success and current_player == Player.WHITE and king_dist_before is not None:
                king_dist_after = self._king_distance_to_closest_escape()
                if king_dist_after is not None and king_dist_before is not None:
                    # If distance decreased
                    if king_dist_after < king_dist_before:
                        reward += self.rewards['KING_CLOSER']

            # ----------------------------
            # E) Check if game ended
            # ----------------------------
            if end_reason:
                info["end_reason"] = end_reason
                winner = self.game.get_winner()
                if winner is None:
                    reward += self.rewards['DRAW']  # draw
                elif winner == current_player:
                    reward += self.rewards['WIN']  # win
                else:
                    reward += self.rewards['LOSS']  # loss
                done = True

        # Step penalty
        reward += self.rewards['STEP_PENALTY']

        self.steps_taken += 1
        if self.steps_taken >= self.move_limit and not done:
            reward += self.rewards['DRAW']  # treat that as a draw
            done = True
            info["end_reason"] = "timeout"

        obs = self._get_observation()
        return obs, reward, done, info

    def _get_observation(self):
        import numpy as np
        board = self.game.board
        obs = np.zeros(self.obs_shape, dtype=np.float32)

        for r in range(9):
            for c in range(9):
                piece = board[r][c]
                if piece == Piece.BLACK:
                    obs[0, r, c] = 1.0
                elif piece == Piece.WHITE:
                    obs[1, r, c] = 1.0
                elif piece == Piece.KING:
                    obs[2, r, c] = 1.0
                elif piece == Piece.CAMP:
                    obs[3, r, c] = 1.0
                elif piece == Piece.ESCAPE:
                    obs[4, r, c] = 1.0
                elif piece == Piece.CASTLE:
                    obs[5, r, c] = 1.0
        
        return obs

    def _king_distance_to_closest_escape(self):
        """
        Return the Manhattan distance from the King to the closest escape tile,
        or None if the king isn't on the board.
        """
        king_pos = None
        for r in range(9):
            for c in range(9):
                if self.game.board[r][c] == Piece.KING:
                    king_pos = (r, c)
                    break
            if king_pos:
                break

        if not king_pos:
            return None  # King is captured or absent

        min_dist = None
        for (er, ec) in self.game.ESCAPE_TILES:
            dist = abs(er - king_pos[0]) + abs(ec - king_pos[1])
            if min_dist is None or dist < min_dist:
                min_dist = dist
        return min_dist


###############################################################################
# 2) ACTION MASKING HELPERS
###############################################################################
def get_valid_action_mask(game: TablutGame) -> np.ndarray:
    """
    Return a boolean array of shape (6561,) indicating which actions are valid
    for the current player in 'game'.
    """
    mask = np.zeros(6561, dtype=bool)
    current_player = game.current_player

    for fr in range(9):
        for fc in range(9):
            piece = game.board[fr][fc]
            # Check ownership
            if current_player == Player.WHITE and piece not in [Piece.WHITE, Piece.KING]:
                continue
            if current_player == Player.BLACK and piece != Piece.BLACK:
                continue

            valid_moves = game.get_valid_moves(fr, fc)  # returns list of (tr, tc)
            from_idx = fr * 9 + fc
            for (tr, tc) in valid_moves:
                to_idx = tr * 9 + tc
                action_id = from_idx * 81 + to_idx
                mask[action_id] = True

    return mask

def select_random_valid_action(game: TablutGame):
    """
    Return a random valid action from the current player's perspective.
    """
    mask = get_valid_action_mask(game)
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        return 0
    return np.random.choice(valid_indices)


class TablutPPONetworkEnhanced(nn.Module):
    def __init__(self, in_channels=6):
        super(TablutPPONetworkEnhanced, self).__init__()
        
        # Increased channel counts and more complex architecture
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Residual blocks
        self.res_block1 = ResidualBlock(32, 64)
        self.res_block2 = ResidualBlock(64, 64)
        self.res_block3 = ResidualBlock(64, 32)
        
        # Final convolution before flattening
        self.conv_final = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn_final = nn.BatchNorm2d(32)
        
        # Number of features after flattening: 32 * 9 * 9 = 2592
        self.bottleneck = nn.Linear(2592, 256)
        
        # Policy heads with additional layers
        self.policy_common = nn.Linear(256, 128)
        self.from_head = nn.Linear(128, 81)
        self.to_head = nn.Linear(128, 81)
        
        # Value head with additional layer
        self.value_hidden = nn.Linear(256, 128)
        self.value_head = nn.Linear(128, 1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual blocks
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        
        # Final convolution
        x = F.relu(self.bn_final(self.conv_final(x)))
        
        # Flatten and bottleneck
        features = x.view(x.size(0), -1)  # (B, 32, 9, 9) -> (B, 2592)
        features = F.relu(self.bottleneck(features))
        features = self.dropout(features)
        
        # Policy branches
        policy_common = F.relu(self.policy_common(features))
        policy_common = self.dropout(policy_common)
        from_logits = self.from_head(policy_common)
        to_logits = self.to_head(policy_common)
        
        # Value branch
        value_hidden = F.relu(self.value_hidden(features))
        value = self.value_head(value_hidden)
        
        return from_logits, to_logits, value


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection with 1x1 conv if dimensions change
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.skip(residual)
        out = F.relu(out)
        
        return out

class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.masks = []  # For action masking
    
    def add(self, state, action, prob, value, reward, done, mask):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.masks.append(mask)
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.masks.clear()
    
    def __len__(self):
        return len(self.states)

def get_valid_from_mask(game: TablutGame) -> np.ndarray:
    """Return a boolean array of shape (81,) indicating which 'from' positions have valid moves."""
    mask = np.zeros(81, dtype=bool)
    current_player = game.current_player

    for fr in range(9):
        for fc in range(9):
            piece = game.board[fr][fc]
            # Check ownership
            if current_player == Player.WHITE and piece not in [Piece.WHITE, Piece.KING]:
                continue
            if current_player == Player.BLACK and piece != Piece.BLACK:
                continue

            valid_moves = game.get_valid_moves(fr, fc)
            if valid_moves:  # If this piece has any valid moves
                from_idx = fr * 9 + fc
                mask[from_idx] = True

    return mask

def get_valid_to_mask(game: TablutGame, from_pos: int) -> np.ndarray:
    """Return a boolean array of shape (81,) indicating valid 'to' positions for the given 'from' position."""
    mask = np.zeros(81, dtype=bool)
    fr, fc = divmod(from_pos, 9)
    
    valid_moves = game.get_valid_moves(fr, fc)
    for tr, tc in valid_moves:
        to_idx = tr * 9 + tc
        mask[to_idx] = True
        
    return mask

def select_action(obs, env, policy_network, device, evaluate=False, temperature=1.0):
    # Get valid from positions mask
    from_mask = get_valid_from_mask(env.game)
    from_mask_tensor = torch.BoolTensor(from_mask).to(device)
    
    # Forward pass through network
    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        from_logits, to_logits, state_value = policy_network(state_tensor)
        
    # Apply temperature and masking to from logits
    from_logits = from_logits.squeeze(0) / temperature
    from_logits[~from_mask_tensor] = float('-inf')
    from_probs = F.softmax(from_logits, dim=0)
    
    # Sample or get best from position
    if evaluate:
        from_pos = torch.argmax(from_probs).item()
    else:
        from_dist = Categorical(from_probs)
        from_pos = from_dist.sample().item()
    
    # Now get valid to positions for this from position
    to_mask = get_valid_to_mask(env.game, from_pos)
    to_mask_tensor = torch.BoolTensor(to_mask).to(device)
    
    # Apply temperature and masking to to logits
    to_logits = to_logits.squeeze(0) / temperature
    to_logits[~to_mask_tensor] = float('-inf')
    to_probs = F.softmax(to_logits, dim=0)
    
    # Sample or get best to position
    if evaluate:
        to_pos = torch.argmax(to_probs).item()
        log_prob = 0  # Not used in evaluation
    else:
        to_dist = Categorical(to_probs)
        to_pos = to_dist.sample().item()
        log_prob = from_dist.log_prob(torch.tensor(from_pos, device=device)) + to_dist.log_prob(torch.tensor(to_pos, device=device))
    
    # Combine into single action for environment
    action = from_pos * 81 + to_pos
    
    if evaluate:
        return action, log_prob, state_value.item(), (from_mask, to_mask)
    else:
        return action, log_prob.item(), state_value.item(), (from_mask, to_mask)

def compute_gae(rewards, values, dones, next_value, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation"""
    advantages = []
    adv = 0
    
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        else:
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        
        adv = delta + gamma * gae_lambda * (1 - dones[t]) * adv
        advantages.insert(0, adv)
        
    returns = [adv + val for adv, val in zip(advantages, values)]
    return advantages, returns

def train_ppo(policy_network, optimizer, memory, device, 
              epochs=4, clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, 
              gamma=0.99, gae_lambda=0.95, batch_size=64):
    """Train policy network using PPO algorithm with mini-batches"""
    # Convert memory to tensors
    states = torch.FloatTensor(np.array(memory.states)).to(device)
    actions = memory.actions  # We'll extract from/to from each action
    old_log_probs = torch.FloatTensor(memory.probs).to(device)
    values = torch.FloatTensor(memory.values).to(device)
    rewards = memory.rewards
    dones = memory.dones
    masks = memory.masks  # Now contains (from_mask, to_mask) tuples
    
    # Extract from and to positions from actions
    from_positions = [action // 81 for action in actions]
    to_positions = [action % 81 for action in actions]
    
    # Compute advantages and returns
    with torch.no_grad():
        _, _, next_value = policy_network(torch.FloatTensor(memory.states[-1]).unsqueeze(0).to(device))
        next_value = next_value.item()
    
    advantages, returns = compute_gae(rewards, values, dones, next_value, gamma=gamma, gae_lambda=gae_lambda)
    advantages = torch.FloatTensor(advantages).to(device)
    
    # Normalize advantages - crucial for training stability
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    returns = torch.FloatTensor(returns).to(device)
    
    # Mini-batch training
    total_loss = 0
    policy_loss_sum = 0
    value_loss_sum = 0
    entropy_sum = 0
    
    # Get total number of samples
    n_samples = len(memory.states)
    
    for _ in range(epochs):
        # Shuffle data for mini-batches
        indices = np.random.permutation(n_samples)
        
        # Process mini-batches
        for start_idx in range(0, n_samples, batch_size):
            end_idx = min(start_idx + batch_size, n_samples)
            mb_indices = indices[start_idx:end_idx]
            
            # Get mini-batch data
            mb_states = states[mb_indices]
            mb_from_positions = torch.LongTensor([from_positions[i] for i in mb_indices]).to(device)
            mb_to_positions = torch.LongTensor([to_positions[i] for i in mb_indices]).to(device)
            mb_old_log_probs = old_log_probs[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = returns[mb_indices]
            mb_from_masks = [masks[i][0] for i in mb_indices]
            mb_to_masks = [masks[i][1] for i in mb_indices]
            
            # Forward pass for this mini-batch
            mb_from_logits, mb_to_logits, mb_values = policy_network(mb_states)
            
            # Apply masks
            for i in range(len(mb_from_logits)):
                mb_from_logits[i][~torch.BoolTensor(mb_from_masks[i]).to(device)] = float('-inf')
                mb_to_logits[i][~torch.BoolTensor(mb_to_masks[i]).to(device)] = float('-inf')
            
            # Convert to probability distributions
            mb_from_dists = [Categorical(logits=mb_from_logits[i]) for i in range(len(mb_from_logits))]
            mb_to_dists = [Categorical(logits=mb_to_logits[i]) for i in range(len(mb_to_logits))]
            
            # Calculate new log probs
            mb_new_from_log_probs = torch.stack([dist.log_prob(mb_from_positions[i]) for i, dist in enumerate(mb_from_dists)])
            mb_new_to_log_probs = torch.stack([dist.log_prob(mb_to_positions[i]) for i, dist in enumerate(mb_to_dists)])
            mb_new_log_probs = mb_new_from_log_probs + mb_new_to_log_probs
            
            # Calculate entropy
            mb_entropy = torch.stack([dist.entropy() for dist in mb_from_dists]).mean() + \
                         torch.stack([dist.entropy() for dist in mb_to_dists]).mean()
            
            # Policy loss with clipping
            mb_ratio = torch.exp(mb_new_log_probs - mb_old_log_probs)
            mb_surr1 = mb_ratio * mb_advantages
            mb_surr2 = torch.clamp(mb_ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * mb_advantages
            mb_policy_loss = -torch.min(mb_surr1, mb_surr2).mean()
            
            # Value loss
            mb_value_loss = F.mse_loss(mb_values.squeeze(-1), mb_returns)
            
            # Combined loss
            mb_loss = mb_policy_loss + value_coef * mb_value_loss - entropy_coef * mb_entropy
            
            # Update policy
            optimizer.zero_grad()
            mb_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=0.5)
            optimizer.step()
            
            # Track metrics
            total_loss += mb_loss.item()
            policy_loss_sum += mb_policy_loss.item()
            value_loss_sum += mb_value_loss.item()
            entropy_sum += mb_entropy.item()
    
    # Compute average metrics
    n_updates = epochs * ((n_samples + batch_size - 1) // batch_size)
    return {
        'total_loss': total_loss / n_updates,
        'policy_loss': policy_loss_sum / n_updates,
        'value_loss': value_loss_sum / n_updates,
        'entropy': entropy_sum / n_updates
    }

def plot_training_metrics(metrics, save_dir="./plots"):
    """Plot and save training metrics charts"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot loss (using proper length)
    plt.figure(figsize=(12, 6))
    loss_episodes = list(range(1, len(metrics['policy_losses']) + 1))
    plt.plot(loss_episodes, metrics['policy_losses'], label='Policy Loss')
    plt.plot(loss_episodes, metrics['value_losses'], label='Value Loss')
    plt.title('Training Losses Over Time')
    plt.xlabel('Training Updates')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    
    # Plot rewards (using all episodes)
    plt.figure(figsize=(12, 6))
    reward_episodes = list(range(1, len(metrics['episode_rewards']) + 1))
    plt.plot(reward_episodes, metrics['episode_rewards'])
    plt.title('Episode Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'reward_plot.png'))
    
    # Plot entropy (same length as losses)
    plt.figure(figsize=(12, 6))
    plt.plot(loss_episodes, metrics['entropies'])
    plt.title('Policy Entropy Over Time')
    plt.xlabel('Training Updates')
    plt.ylabel('Entropy')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'entropy_plot.png'))
    
    print(f"Training plots saved to {save_dir}")

def evaluate_vs_random(policy_network, agent_color, episodes=10, device="cpu", move_limit=150, custom_rewards=None):
    """
    Plays the agent_color side vs. random (which also picks only valid moves).
    Returns fraction of games won by 'agent_color'.
    """
    wins = 0
    end_reasons = []
    for _ in range(episodes):
        env = TablutEnv(move_limit=move_limit, rewards=custom_rewards)
        obs = env.reset()
        done = False
        info = {}
        while not done:
            current_player = env.game.current_player
            if current_player == agent_color:
                # Agent picks a greedy valid action using policy network
                action, _, _, _ = select_action(obs, env, policy_network, device, evaluate=True)
            else:
                # Opponent picks random valid
                action = select_random_valid_action(env.game)

            obs, reward, done, info = env.step(action)

        winner = env.game.get_winner()
        if winner == agent_color:
            wins += 1
        if "end_reason" in info:
            end_reasons.append(info["end_reason"])

    # Log end reasons with color label
    color_name = "White" if agent_color == Player.WHITE else "Black"
    print(f"  - {color_name} agent results:")
    for reason in set(end_reasons):
        count = end_reasons.count(reason)
        print(f"    * End reason '{reason}': {count}/{episodes}")

    return wins / episodes

def sample_batch(memory, batch_size):
    """Sample random transitions from memory"""
    if len(memory) <= batch_size:
        return memory
    
    indices = np.random.choice(len(memory.states), batch_size, replace=False)
    return {
        'states': [memory.states[i] for i in indices],
        'actions': [memory.actions[i] for i in indices],
        'probs': [memory.probs[i] for i in indices],
        'values': [memory.values[i] for i in indices],
        'rewards': [memory.rewards[i] for i in indices],
        'dones': [memory.dones[i] for i in indices],
        'masks': [memory.masks[i] for i in indices]
    }

def count_black_attackers_near_king(game):
    """Count how many black pieces are adjacent to the king"""
    # Find king position
    king_position = None
    for i in range(9):
        for j in range(9):
            if game.board[i][j] == Piece.KING:
                king_position = (i, j)
                break
        if king_position:
            break
    
    if not king_position:
        return 0  # King already captured
    
    # Check adjacent squares for black pieces
    black_adjacent = 0
    row, col = king_position
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < 9 and 0 <= new_col < 9:
            if game.board[new_row][new_col] == Piece.BLACK:
                black_adjacent += 1
    
    return black_adjacent

def main():
    # Hyperparameters
    LR = 3e-4
    GAMMA = 0.99
    GAE_LAMBDA = 0.9
    CLIP_RATIO = 0.2
    VALUE_COEF = 0.5
    INITIAL_ENTROPY = 0.1
    FINAL_ENTROPY = 0.001
    MAX_EPISODES = 3000
    MAX_STEPS_PER_EPISODE = 150
    UPDATE_FREQ = 512
    PPO_EPOCHS = 4
    EVAL_FREQ = 100
    SAVE_FREQ = 500
    MIN_BATCH_SIZE = 512
    
    # Choose which agent to train
    AGENT_COLOR = Player.WHITE  # Change to Player.WHITE to train white agent
    
    # Custom rewards with KING_THREATENED reward
    custom_rewards = {
        'CAPTURE_PIECE': 0.0,
        'KING_CLOSER': 0.0,
        'KING_THREATENED': 0.0, # Disable
        'WIN': 10.0,            # Or 1.0
        'INVALID_MOVE': 0.0,    # Assuming masking works
        'STEP_PENALTY': -0.01,   # Small penalty
        'DRAW': 0.0,             # Usually 0 for draw
        'LOSS': -10.0,           # Or -1.0 (Symmetric to WIN)
    }

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = TablutEnv(move_limit=MAX_STEPS_PER_EPISODE, rewards=custom_rewards)
    
    # Create policy network
    policy_network = TablutPPONetworkEnhanced().to(device)
    
    # Count and print trainable parameters
    total_params = sum(p.numel() for p in policy_network.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    print(f"Training as {'BLACK' if AGENT_COLOR == Player.BLACK else 'WHITE'} player")
    
    optimizer = optim.Adam(policy_network.parameters(), lr=LR)
    
    # Create memory
    memory = PPOMemory()
    
    # Training metrics
    metrics = {
        'episode_rewards': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
        'win_rates': []
    }
    
    # Start training
    total_steps = 0
    total_steps_since_update = 0
    
    INITIAL_TEMP = 1.0
    FINAL_TEMP = 0.1
    
    # Add this at the beginning of the main function (with other variables)
    best_win_rate = 0.0
    
    for episode in range(MAX_EPISODES):
        # Decay learning rate
        if episode < 500:
            lr = LR
        else:
            lr = LR * max(0.1, (1 - (episode-500)/MAX_EPISODES))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        obs = env.reset()
        done = False
        episode_reward = 0
        
        # Track previous number of black pieces adjacent to king for reward calculation
        prev_black_attackers = count_black_attackers_near_king(env.game)
        
        while not done:
            # Temperature calculation
            temp = max(FINAL_TEMP, INITIAL_TEMP * (1 - episode/MAX_EPISODES))
            
            current_player = env.game.current_player
            
            if current_player == AGENT_COLOR:
                # Agent makes a move and learns using its policy network
                action, log_prob, value, valid_mask = select_action(obs, env, policy_network, device, temperature=temp)
                next_obs, reward, done, info = env.step(action)
                
                # Store in memory with proper log_prob and value
                memory.add(obs, action, log_prob, value, reward, done, valid_mask)
                
                # Track episode reward
                episode_reward += reward
                total_steps += 1
                total_steps_since_update += 1
            else:
                # Opponent makes a random move
                action = select_random_valid_action(env.game)
                next_obs, reward, done, info = env.step(action)
                
                # Don't count opponent's rewards in episode total
                # Don't store opponent's actions in memory
            
            # Update state
            obs = next_obs
        
        # Train when we've collected enough steps
        if total_steps_since_update >= UPDATE_FREQ and len(memory) >= MIN_BATCH_SIZE:
            # Train PPO with accumulated experience
            entropy_coef = max(FINAL_ENTROPY, INITIAL_ENTROPY * (1 - episode/MAX_EPISODES))
            training_stats = train_ppo(
                policy_network=policy_network,
                optimizer=optimizer,
                memory=memory,
                device=device,
                epochs=PPO_EPOCHS,
                clip_ratio=CLIP_RATIO,
                value_coef=VALUE_COEF,
                entropy_coef=entropy_coef,
                gamma=GAMMA,
                gae_lambda=GAE_LAMBDA,
                batch_size=64
            )
            
            # Store metrics
            metrics['policy_losses'].append(training_stats['policy_loss'])
            metrics['value_losses'].append(training_stats['value_loss'])
            metrics['entropies'].append(training_stats['entropy'])
            
            # Reset step counter and clear memory
            total_steps_since_update = 0
            memory.clear()
        
        # Store episode reward
        metrics['episode_rewards'].append(episode_reward)
        
        # Log progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(metrics['episode_rewards'][-10:])
            print(f"Episode {episode+1}/{MAX_EPISODES}, Avg Reward: {avg_reward:.3f}")
            
            if metrics['policy_losses']:
                avg_policy_loss = np.mean(metrics['policy_losses'][-10:])
                avg_value_loss = np.mean(metrics['value_losses'][-10:])
                avg_entropy = np.mean(metrics['entropies'][-10:])
                print(f"  Policy Loss: {avg_policy_loss:.6f}, Value Loss: {avg_value_loss:.6f}, Entropy: {avg_entropy:.6f}")
        
        # Evaluation
        if (episode + 1) % EVAL_FREQ == 0:
            print("Evaluating performance vs. random player...")
            win_rate = evaluate_vs_random(policy_network, AGENT_COLOR, episodes=100, device=device, 
                                         move_limit=MAX_STEPS_PER_EPISODE, custom_rewards=custom_rewards)
            metrics['win_rates'].append(win_rate)
            print(f"  Win Rate: {win_rate:.2f}")
            
            # Save model only if it's the best win rate so far
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                color_name = "black" if AGENT_COLOR == Player.BLACK else "white"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Format with win rate (as percentage)
                win_rate_str = f"wr{int(win_rate * 100)}"
                
                save_dir = os.path.join("model", f"ppo_{color_name}_{timestamp}")
                os.makedirs(save_dir, exist_ok=True)
                
                model_path = os.path.join(save_dir, f"tablut_ppo_{color_name}_{win_rate_str}_ep{episode+1}.pth")
                torch.save(policy_network.state_dict(), model_path)
                print(f"New best model saved with win rate {win_rate:.2f} to {model_path}")
                
                # Plot and save metrics
                plot_training_metrics(metrics, save_dir=os.path.join(save_dir, "plots"))

if __name__ == "__main__":
    main()