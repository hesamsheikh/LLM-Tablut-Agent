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
from rl_agent import TablutEnv, get_valid_action_mask, select_random_valid_action

class TablutPPONetwork(nn.Module):
    def __init__(self, in_channels=7, num_actions=81*81):
        super(TablutPPONetwork, self).__init__()
        # Shared feature extractor (CNN)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        
        # Policy head - outputs action probabilities
        self.policy_head = nn.Linear(2592, num_actions)
        
        # Value head - estimates state value
        self.value_head = nn.Linear(2592, 1)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)
    
    def forward(self, x):
        # Extract features
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        features = x.view(x.size(0), -1)  # Flatten: (B, 32, 9, 9) -> (B, 2592)
        
        # Policy: output logits (pre-softmax)
        policy_logits = self.policy_head(features)
        
        # Value: estimate state value
        value = self.value_head(features)
        
        return policy_logits, value

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

def select_action(obs, env, policy_network, device, evaluate=False, temperature=1.0):
    # Get valid action mask
    valid_mask = get_valid_action_mask(env.game)
    valid_mask_tensor = torch.BoolTensor(valid_mask).to(device)
    
    # Forward pass through network
    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
    with torch.no_grad():
        policy_logits, state_value = policy_network(state_tensor)
        
    # Apply temperature to logits before softmax for more exploration
    policy_logits = policy_logits.squeeze(0) / temperature
    policy_logits[~valid_mask_tensor] = float('-inf')
    
    # Convert to probabilities
    action_probs = F.softmax(policy_logits, dim=0)
    
    # Sample from probability distribution or take best action
    if evaluate:
        action = torch.argmax(action_probs).item()
        return action, 0, state_value.item(), valid_mask
    else:
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        return action.item(), log_prob.item(), state_value.item(), valid_mask

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
    actions = torch.LongTensor(memory.actions).to(device)
    old_log_probs = torch.FloatTensor(memory.probs).to(device)
    values = torch.FloatTensor(memory.values).to(device)
    rewards = memory.rewards
    dones = memory.dones
    masks = [torch.BoolTensor(mask).to(device) for mask in memory.masks]
    
    # Compute advantages and returns
    with torch.no_grad():
        _, next_value = policy_network(torch.FloatTensor(memory.states[-1]).unsqueeze(0).to(device))
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
            mb_actions = actions[mb_indices]
            mb_old_log_probs = old_log_probs[mb_indices]
            mb_advantages = advantages[mb_indices]
            mb_returns = returns[mb_indices]
            mb_masks = [masks[i] for i in mb_indices]
            
            # Forward pass for this mini-batch
            mb_policy_logits, mb_values = policy_network(mb_states)
            
            # Apply action masks
            for i in range(len(mb_policy_logits)):
                mb_policy_logits[i][~mb_masks[i]] = float('-inf')
            
            # Convert to probability distributions
            mb_action_dists = [Categorical(logits=mb_policy_logits[i]) for i in range(len(mb_policy_logits))]
            
            # Calculate new log probs
            mb_new_log_probs = torch.stack([dist.log_prob(mb_actions[i]) for i, dist in enumerate(mb_action_dists)])
            
            # Calculate entropy
            mb_entropy = torch.stack([dist.entropy() for dist in mb_action_dists]).mean()
            
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
    
    # Compute average metrics (dividing by epochs × number of mini-batches)
    n_updates = epochs * ((n_samples + batch_size - 1) // batch_size)  # Ceiling division
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

def evaluate_vs_random(policy_network, agent_color, episodes=10, device="cpu"):
    """
    Plays the agent_color side vs. random (which also picks only valid moves).
    Returns fraction of games won by 'agent_color'.
    """
    wins = 0
    end_reasons = []
    for _ in range(episodes):
        env = TablutEnv(move_limit=40)
        obs = env.reset()
        done = False
        info = {}
        while not done:
            current_player = env.game.current_player
            if current_player == agent_color:
                # Agent picks a greedy valid action using policy network
                valid_mask = get_valid_action_mask(env.game)
                state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                with torch.no_grad():
                    policy_logits, _ = policy_network(state_t)
                    policy_logits = policy_logits.squeeze(0)
                    # Apply mask
                    policy_logits[~torch.BoolTensor(valid_mask).to(device)] = float('-inf')
                    action_probs = F.softmax(policy_logits, dim=0)
                    action = torch.argmax(action_probs).item()
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
    INITIAL_ENTROPY = 0.05
    FINAL_ENTROPY = 0.01
    MAX_EPISODES = 3000
    MAX_STEPS_PER_EPISODE = 150
    UPDATE_FREQ = 1024
    PPO_EPOCHS = 4
    EVAL_FREQ = 100
    SAVE_FREQ = 500
    MIN_BATCH_SIZE = 512
    
    # Custom rewards with KING_THREATENED reward
    custom_rewards = {
        'CAPTURE_PIECE': 0.5,
        'KING_CLOSER': 0.5,
        'KING_THREATENED': 0.7,  # New reward for black pieces adjacent to king
        'WIN': 10.0,               
        'INVALID_MOVE': -0.2,      
        'STEP_PENALTY': -0.02,
        'DRAW': 0,
        'LOSS': -10.0,
    }

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create environment
    env = TablutEnv(move_limit=MAX_STEPS_PER_EPISODE, rewards=custom_rewards)
    
    # Create policy network
    policy_network = TablutPPONetwork().to(device)
    
    # Count and print trainable parameters
    total_params = sum(p.numel() for p in policy_network.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
    
    optimizer = optim.Adam(policy_network.parameters(), lr=LR)
    
    # Create memory
    memory = PPOMemory()
    
    # Training metrics
    metrics = {
        'episode_rewards': [],
        'policy_losses': [],
        'value_losses': [],
        'entropies': [],
        'eval_white_winrates': [],
        'eval_black_winrates': []
    }
    
    # Start training
    total_steps = 0
    
    # Black curriculum configuration
    INITIAL_FOCUS_BLACK = True
    CURRICULUM_EPISODES = 500  # Give Black a longer curriculum since it's harder to learn
    
    INITIAL_TEMP = 1.5
    FINAL_TEMP = 0.2
    
    total_steps_since_update = 0
    
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
        
        # Store previous state to calculate king threat differences
        prev_black_attackers = count_black_attackers_near_king(env.game)
        
        if INITIAL_FOCUS_BLACK and episode < CURRICULUM_EPISODES:
            # Black curriculum phase
            while not done:
                # Temperature calculation
                temp = max(FINAL_TEMP, INITIAL_TEMP * (1 - episode/500))
                
                current_player = env.game.current_player
                
                if current_player == Player.WHITE:
                    # Let White make a random move (not learning from it)
                    action = select_random_valid_action(env.game)
                    next_obs, reward, done, info = env.step(action)
                    # Don't add White's action to memory
                else:
                    # Black's turn - agent makes move and learns
                    action, log_prob, value, valid_mask = select_action(obs, env, policy_network, device, temperature=temp)
                    next_obs, reward, done, info = env.step(action)
                    
                    # Calculate king threat reward after move
                    current_black_attackers = count_black_attackers_near_king(env.game)
                    attackers_diff = current_black_attackers - prev_black_attackers
                    if attackers_diff > 0:
                        # Black gets positive reward for increasing threat to king
                        king_reward = attackers_diff * env.rewards['KING_THREATENED']
                        reward += king_reward
                    
                    memory.add(obs, action, log_prob, value, reward, done, valid_mask)
                    prev_black_attackers = current_black_attackers
                
                obs = next_obs
                episode_reward += reward
                total_steps += 1
                total_steps_since_update += 1
        else:
            # Normal self-play phase
            while not done:
                # Calculate temperature
                temp = max(FINAL_TEMP, INITIAL_TEMP * (1 - episode/500))
                action, log_prob, value, valid_mask = select_action(obs, env, policy_network, device, temperature=temp)
                
                # Execute action
                next_obs, reward, done, info = env.step(action)
                
                # Calculate king threat reward after move
                current_black_attackers = count_black_attackers_near_king(env.game)
                attackers_diff = current_black_attackers - prev_black_attackers
                
                # Apply reward based on player
                if env.game.current_player == Player.BLACK:
                    if attackers_diff > 0:
                        # Black gets positive reward for increasing threat to king
                        king_reward = attackers_diff * env.rewards['KING_THREATENED']
                        reward += king_reward
                else:  # White
                    if attackers_diff > 0:
                        # White gets negative reward for allowing increased threat to king
                        king_reward = -attackers_diff * env.rewards['KING_THREATENED']
                        reward += king_reward
                
                # Store in memory
                memory.add(obs, action, log_prob, value, reward, done, valid_mask)
                
                obs = next_obs
                episode_reward += reward
                total_steps += 1
                total_steps_since_update += 1
                
                # Update previous state
                prev_black_attackers = current_black_attackers
        
        # Train when we've collected enough steps
        if total_steps_since_update >= UPDATE_FREQ and len(memory) >= MIN_BATCH_SIZE:
            # Train PPO with accumulated experience
            entropy_coef = max(FINAL_ENTROPY, INITIAL_ENTROPY * (1 - episode / (MAX_EPISODES * 1.5))) # Decay over 1.5x the episodes
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
            win_rate_white = evaluate_vs_random(policy_network, Player.WHITE, episodes=20, device=device)
            win_rate_black = evaluate_vs_random(policy_network, Player.BLACK, episodes=20, device=device)
            metrics['eval_white_winrates'].append(win_rate_white)
            metrics['eval_black_winrates'].append(win_rate_black)
            print(f"  White Win Rate: {win_rate_white:.2f}, Black Win Rate: {win_rate_black:.2f}")
        
        # Save model
        if (episode + 1) % SAVE_FREQ == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join("model", "ppo_" + timestamp)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"tablut_ppo_ep{episode+1}.pth")
            torch.save(policy_network.state_dict(), model_path)
            print(f"Model saved to {model_path}")
            
            # Plot and save metrics
            plot_training_metrics(metrics, save_dir=os.path.join(save_dir, "plots"))

if __name__ == "__main__":
    main()