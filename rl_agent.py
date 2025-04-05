#########################################################
# train_dqn.py - with Action Masking and Additional Reward Shaping
# Make sure tablut.py (with TablutGame class) is present
# in the same directory. 
#########################################################

import os
import random
import numpy as np
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Import your existing TablutGame code
from tablut import TablutGame, Player, Piece

###############################################################################
# 0) REWARD CONFIGURATION
###############################################################################
DEFAULT_REWARDS = {
    # Positive rewards
    'CAPTURE_PIECE': 0.02,        # For capturing an opponent piece
    'KING_CLOSER': 0.03,           # When the king moves closer to escape
    'WIN': 1.0,                   # When the current player wins
    
    # Negative rewards
    'INVALID_MOVE': -0.02,        # For attempting an invalid move
    'STEP_PENALTY': -0.001,       # Small penalty for each step (encourages faster wins)
    'DRAW': -0.8,                 # For a draw or timeout
    'LOSS': -1.0,                 # When the current player loses
}

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
        # Observations: [Black, White, King, Camp, Escape, Castle, CurrentPlayer]
        self.obs_shape = (7, 9, 9)
        
        # Initialize rewards, use defaults if not specified
        self.rewards = DEFAULT_REWARDS.copy()
        if rewards:
            self.rewards.update(rewards)
            
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
        
        # Channel 6 => current player
        if self.game.current_player == Player.WHITE:
            obs[6, :, :] = 1.0
        else:
            obs[6, :, :] = 0.0

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

def select_action(obs, env: TablutEnv, dqn, epsilon: float, king_selection_counter=None):
    """
    Epsilon-greedy selection with action masking.
    Optionally tracks King piece selection.
    """
    valid_mask = get_valid_action_mask(env.game)
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) == 0:
        # no valid moves => pick something arbitrary
        return 0

    # Epsilon step
    if random.random() < epsilon:
        # random among valid
        action = np.random.choice(valid_indices)
    else:
        # greedy among valid
        state_t = torch.FloatTensor(obs).unsqueeze(0)  # shape (1,7,9,9)
        with torch.no_grad():
            q_values = dqn(state_t).squeeze(0).numpy()  # shape (6561,)
        # mask out invalid
        q_values[~valid_mask] = -1e9
        action = int(np.argmax(q_values))
    
    # Track King selection if counter provided and current player is White
    if king_selection_counter is not None and env.game.current_player == Player.WHITE:
        from_index = action // 81
        fr, fc = divmod(from_index, 9)
        piece = env.game.board[fr][fc]
        if piece == Piece.KING:
            king_selection_counter['king'] += 1
        elif piece == Piece.WHITE:
            king_selection_counter['soldier'] += 1

    return action

def select_random_valid_action(game: TablutGame):
    """
    Return a random valid action from the current player's perspective.
    """
    mask = get_valid_action_mask(game)
    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        return 0
    return np.random.choice(valid_indices)

###############################################################################
# 3) SIMPLE CNN-BASED DQN
###############################################################################
class DQN(nn.Module):
    def __init__(self, in_channels=7, num_actions=81*81):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        # shape after conv3: (32,9,9) => 32*9*9 = 2592
        self.fc1 = nn.Linear(2592, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

###############################################################################
# 4) REPLAY BUFFER
###############################################################################
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards_, next_states, dones_ = zip(*samples)
        return np.array(states), actions, rewards_, np.array(next_states), dones_

    def __len__(self):
        return len(self.buffer)

###############################################################################
# 5) EVALUATION VS RANDOM OPPONENT
###############################################################################
def evaluate_vs_random(dqn, agent_color, episodes=10):
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
                # Agent picks a greedy valid action
                valid_mask = get_valid_action_mask(env.game)
                state_t = torch.FloatTensor(obs).unsqueeze(0)
                with torch.no_grad():
                    q_values = dqn(state_t).squeeze(0).numpy()
                q_values[~valid_mask] = -1e9
                action = int(np.argmax(q_values))
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

###############################################################################
# 6) TRAINING LOOP
###############################################################################
def plot_training_metrics(episode_losses, episode_q_means, episode_q_mins, episode_q_maxs, 
                          episode_rewards, save_dir="./plots"):
    """
    Plot and save training metrics charts.
    """
    os.makedirs(save_dir, exist_ok=True)
    episodes = list(range(1, len(episode_losses) + 1))
    
    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_plot.png'))
    
    # Plot Q-values
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_q_means, label='Mean Q')
    plt.plot(episodes, episode_q_mins, label='Min Q')
    plt.plot(episodes, episode_q_maxs, label='Max Q')
    plt.title('Q-values Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Q-value')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'q_values_plot.png'))
    
    # Plot rewards
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, episode_rewards)
    plt.title('Episode Rewards Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'reward_plot.png'))
    
    print(f"Training plots saved to {save_dir}")

def main():
    # Hyperparameters
    LR = 1e-5
    GAMMA = 0.90
    BATCH_SIZE = 8 
    REPLAY_SIZE = 20000
    MIN_REPLAY_SIZE = 1000
    EPS_START = 1.0
    EPS_END = 0.01
    EPS_DECAY = 100000  # steps
    TARGET_UPDATE_FREQ = 1000
    MAX_EPISODES = 200
    MAX_STEPS_PER_EPISODE = 300
    SAVE_EVERY = 200  # model save frequency
    
    # Custom rewards (optional)
    custom_rewards = {
        'CAPTURE_PIECE': 0.07,
        'KING_CLOSER': 0.1,
        'WIN': 1.0,
        'INVALID_MOVE': -0.05,
        'STEP_PENALTY': -0.005,
        'DRAW': -2.0,
        'LOSS': -1.0,
    }

    # Create environment and networks
    env = TablutEnv(move_limit=40, rewards=custom_rewards)
    dqn = DQN()
    trainable_params = sum(p.numel() for p in dqn.parameters() if p.requires_grad)
    print(f"Number of trainable parameters in DQN: {trainable_params:,}")
    dqn_target = DQN()
    dqn_target.load_state_dict(dqn.state_dict())

    optimizer = optim.Adam(dqn.parameters(), lr=LR)
    replay_buffer = ReplayBuffer(REPLAY_SIZE)

    epsilon = EPS_START
    eps_decay_step = (EPS_START - EPS_END) / EPS_DECAY

    global_step = 0
    episode_rewards = []
    
    # Tracking metrics
    king_piece_selection = {'king': 0, 'soldier': 0}
    game_end_reasons = []

    # Loss and Q-value tracking
    loss_history = deque(maxlen=100)
    q_mean_history = deque(maxlen=100)
    q_max_history = deque(maxlen=100)
    q_min_history = deque(maxlen=100)

    # Tracking metrics for plotting
    all_episode_losses = []
    all_episode_q_means = []
    all_episode_q_mins = []
    all_episode_q_maxs = []

    # Pre-fill replay buffer with random valid moves
    obs = env.reset()
    mask = get_valid_action_mask(env.game)

    for _ in range(MIN_REPLAY_SIZE):
        action = select_random_valid_action(env.game)
        next_obs, reward, done, info = env.step(action)
        replay_buffer.push(obs, action, reward, next_obs, done)
        obs = next_obs
        if done:
            obs = env.reset()

    # TRAINING LOOP
    for episode in range(MAX_EPISODES):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        episode_captures = 0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            steps += 1
            global_step += 1

            # Epsilon-greedy with action masking
            action = select_action(obs, env, dqn, epsilon, king_piece_selection)

            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # Track captures
            if "captured_pieces" in info:
                episode_captures += info["captured_pieces"]
                
            # Track game end reasons
            if done and "end_reason" in info:
                game_end_reasons.append(info["end_reason"])

            # Store transition
            replay_buffer.push(obs, action, reward, next_obs, done)
            obs = next_obs

            # Sample from replay buffer & learn
            states, actions, rewards_, next_states, dones_ = replay_buffer.sample(BATCH_SIZE)
            states_t = torch.FloatTensor(states)
            actions_t = torch.LongTensor(actions)
            rewards_t = torch.FloatTensor(rewards_)
            next_states_t = torch.FloatTensor(next_states)
            dones_t = torch.FloatTensor(dones_)

            # Current Q-values
            q_vals = dqn(states_t)             # (B, 6561)
            assert q_vals.shape == (BATCH_SIZE, 6561), f"Q shape is {q_vals.shape}"
            q_action = q_vals.gather(1, actions_t.unsqueeze(1)).squeeze(1)  # (B,)
            
            # Track Q-value stats (new)
            with torch.no_grad():
                q_max = q_vals.max(dim=1)[0]
                q_mean = q_max.mean().item()
                q_max_val = q_max.max().item()
                q_min_val = q_max.min().item()

            # Target Q-values
            with torch.no_grad():
                # 1) Use the current (online) network dqn to choose the best next action
                next_action = dqn(next_states_t).argmax(dim=1, keepdim=True)  # shape (B,1)

                # 2) Use the target network dqn_target to evaluate that action
                next_q = dqn_target(next_states_t).gather(1, next_action).squeeze(1)  # shape (B,)

                # 3) Build the target
                target_q = rewards_t + GAMMA * (1 - dones_t) * next_q

            loss = F.smooth_l1_loss(q_action, target_q)
            # Track loss value (new)
            loss_val = loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dqn.parameters(), max_norm=10.0)
            optimizer.step()

            # Store metrics
            loss_history.append(loss_val)
            q_mean_history.append(q_mean)
            q_max_history.append(q_max_val)
            q_min_history.append(q_min_val)

            # Update target net
            if global_step % TARGET_UPDATE_FREQ == 0:
                dqn_target.load_state_dict(dqn.state_dict())

            # Decay epsilon
            if epsilon > EPS_END:
                epsilon -= eps_decay_step

        episode_rewards.append(total_reward)

        # Store episode metrics for plotting
        if loss_history:
            all_episode_losses.append(sum(loss_history) / len(loss_history))
            all_episode_q_means.append(sum(q_mean_history) / len(q_mean_history))
            all_episode_q_mins.append(sum(q_min_history) / len(q_min_history))
            all_episode_q_maxs.append(sum(q_max_history) / len(q_max_history))

        # Logging & Evaluation
        if (episode+1) % 10 == 0:
            avg_last_10 = np.mean(episode_rewards[-10:])
            print(f"[Episode {episode+1}] AvgReward (last 10): {avg_last_10:.3f}, Epsilon: {epsilon:.3f}")
            
            # New: Report loss and Q-value statistics
            if loss_history:
                avg_loss = sum(loss_history) / len(loss_history)
                avg_q_mean = sum(q_mean_history) / len(q_mean_history)
                avg_q_max = sum(q_max_history) / len(q_max_history)
                avg_q_min = sum(q_min_history) / len(q_min_history)
                print(f"  - Training metrics: Loss={avg_loss:.6f}, Q-values: mean={avg_q_mean:.3f}, min={avg_q_min:.3f}, max={avg_q_max:.3f}")
            
            print(f"  - Captures this episode: {episode_captures}")
            # print(f"  - King vs Soldier selection (cumulative): King={king_piece_selection['king']}, Soldier={king_piece_selection['soldier']}")
            
            # Log game end reasons from last 10 episodes
            recent_end_reasons = game_end_reasons[-10:]
            for reason in set(recent_end_reasons):
                if reason:  # Not None
                    count = recent_end_reasons.count(reason)
                    print(f"  - End reason '{reason}': {count}/10 recent games")
            
            # Evaluate vs. random: 10 games as White, 10 games as Black
            print("Evaluation vs. Random:")
            win_rate_white = evaluate_vs_random(dqn, Player.WHITE, episodes=20)
            win_rate_black = evaluate_vs_random(dqn, Player.BLACK, episodes=20)
            print(f"  - WhiteWinRate: {win_rate_white:.2f}, BlackWinRate: {win_rate_black:.2f}")
            # rewards = [b[2] for b in replay_buffer.buffer]
            # print(f"Average reward: {sum(rewards)/len(rewards)}")

        # Save checkpoint
        if (episode+1) % SAVE_EVERY == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join("model", timestamp)
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(save_dir, f"dqn_ep{episode+1}.pth")
            torch.save(dqn.state_dict(), model_path)
            print(f"[Checkpoint] Saved model to {model_path}")

    # After training is complete, plot and save training metrics
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_dir = os.path.join("plots", timestamp)
    plot_training_metrics(
        all_episode_losses, 
        all_episode_q_means, 
        all_episode_q_mins,
        all_episode_q_maxs,
        episode_rewards,
        save_dir=plot_dir
    )

if __name__ == "__main__":
    main()
