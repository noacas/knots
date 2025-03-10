import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.nn import MultiheadAttention

from braid_env import BraidEnvironment


# Check if MPS is available
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("MPS device not found, using CPU")


# BraidAttentionPolicy Network
class BraidAttentionPolicy(nn.Module):
    def __init__(self, d_model, num_heads, output_dim):
        super(BraidAttentionPolicy, self).__init__()

        self.d_model = d_model

        # Embedding layers for current and target braids
        self.current_embedding = nn.Linear(1, d_model)
        self.target_embedding = nn.Linear(1, d_model)

        # Positional encoding
        self.position_encoding = nn.Parameter(torch.randn(1, 100, d_model))  # Max length 100

        # Self-attention for current braid
        self.self_attention = MultiheadAttention(d_model, num_heads)

        # Cross-attention between current and target braids
        self.cross_attention = MultiheadAttention(d_model, num_heads)

        # Feed-forward layers
        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Output layer
        self.output = nn.Linear(d_model, output_dim)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Split input into current and target braids
        batch_size = x.size(0)
        seq_len = x.size(1) // 2
        current_braid = x[:, :seq_len].unsqueeze(-1)
        target_braid = x[:, seq_len:].unsqueeze(-1)

        # Embed current and target braids
        current = self.current_embedding(current_braid)
        target = self.target_embedding(target_braid)

        # Add positional encoding
        current = current + self.position_encoding[:, :seq_len, :]
        target = target + self.position_encoding[:, :seq_len, :]

        # Self-attention on current braid
        self_attn_output, _ = self.self_attention(current, current, current)
        current = self.norm1(current + self.dropout(self_attn_output))

        # Cross-attention between current and target braids
        cross_attn_output, _ = self.cross_attention(current, target, target)
        current = self.norm2(current + self.dropout(cross_attn_output))

        # Feed-forward
        ff_output = self.ff2(F.relu(self.ff1(current)))
        current = self.norm3(current + self.dropout(ff_output))

        # Pool across sequence dimension and project to output dimension
        current = current.mean(dim=1)
        output = self.output(current)

        return F.softmax(output, dim=-1)


# BraidAttentionValue Network
class BraidAttentionValue(nn.Module):
    def __init__(self, d_model, num_heads):
        super(BraidAttentionValue, self).__init__()

        self.d_model = d_model

        # Embedding layers for current and target braids
        self.current_embedding = nn.Linear(1, d_model)
        self.target_embedding = nn.Linear(1, d_model)

        # Positional encoding
        self.position_encoding = nn.Parameter(torch.randn(1, 100, d_model))  # Max length 100

        # Self-attention for current braid
        self.self_attention = MultiheadAttention(d_model, num_heads)

        # Cross-attention between current and target braids
        self.cross_attention = MultiheadAttention(d_model, num_heads)

        # Feed-forward layers
        self.ff1 = nn.Linear(d_model, d_model * 4)
        self.ff2 = nn.Linear(d_model * 4, d_model)

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Output layer
        self.output = nn.Linear(d_model, 1)

        # Dropout
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Split input into current and target braids
        batch_size = x.size(0)
        seq_len = x.size(1) // 2
        current_braid = x[:, :seq_len].unsqueeze(-1)
        target_braid = x[:, seq_len:].unsqueeze(-1)

        # Embed current and target braids
        current = self.current_embedding(current_braid)
        target = self.target_embedding(target_braid)

        # Add positional encoding
        current = current + self.position_encoding[:, :seq_len, :]
        target = target + self.position_encoding[:, :seq_len, :]

        # Self-attention on current braid
        self_attn_output, _ = self.self_attention(current, current, current)
        current = self.norm1(current + self.dropout(self_attn_output))

        # Cross-attention between current and target braids
        cross_attn_output, _ = self.cross_attention(current, target, target)
        current = self.norm2(current + self.dropout(cross_attn_output))

        # Feed-forward
        ff_output = self.ff2(F.relu(self.ff1(current)))
        current = self.norm3(current + self.dropout(ff_output))

        # Pool across sequence dimension and project to output dimension
        current = current.mean(dim=1)
        output = self.output(current)

        return output


class TRPOAgent:
    def __init__(self, env, policy_network, value_network, gamma=0.99, lam=0.95,
                 max_param_change=0.01, backtrack_iters=10, backtrack_coeff=0.8,
                 max_timesteps=1000, vine_bonus_coeff=0.01,
                 vine_batch_size=64, vine_epochs=5):
        self.env = env
        self.policy = policy_network.to(device)
        self.value = value_network.to(device)
        self.gamma = gamma
        self.lam = lam
        self.max_param_change = max_param_change  # Maximum parameter change magnitude
        self.backtrack_iters = backtrack_iters
        self.backtrack_coeff = backtrack_coeff
        self.max_timesteps = max_timesteps

        # VINE parameters
        self.vine_bonus_coeff = vine_bonus_coeff
        self.vine_batch_size = vine_batch_size
        self.vine_epochs = vine_epochs

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=1e-3)

        # For tracking state visitation
        self.state_visitation_count = {}

    def _get_state_key(self, state):
        # Convert state to a hashable representation using tensor hash
        if isinstance(state, torch.Tensor):
            state_tuple = tuple(state.cpu().tolist())
        else:
            state_tuple = tuple(state)
        return state_tuple

    def get_flat_params(self):
        """Get current policy parameters as a single flat vector"""
        params = []
        for param in self.policy.parameters():
            params.append(param.data.view(-1))
        return torch.cat(params).to(device)

    def set_flat_params(self, flat_params):
        """Set policy parameters from a single flat vector"""
        idx = 0
        for param in self.policy.parameters():
            param_size = param.numel()
            param.data.copy_(flat_params[idx:idx + param_size].view(param.shape))
            idx += param_size

    def compute_vine_bonus(self, state):
        # Compute exploration bonus based on state visitation count
        state_key = self._get_state_key(state)
        count = self.state_visitation_count.get(state_key, 0)
        # Bonus is inversely proportional to the square root of visitation count
        if count == 0:
            return 1.0  # Maximum bonus for unseen states
        else:
            return 1.0 / torch.sqrt(torch.tensor(count, dtype=torch.float32, device=device))

    def collect_trajectory(self):
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []

        state = self.env.reset()
        done = False
        total_reward = 0
        timesteps = 0
        found_transformations = False

        while not done and timesteps < self.max_timesteps:
            state_tensor = state.unsqueeze(0).to(device)
            # Get action probabilities from policy
            with torch.no_grad():
                action_probs = self.policy(state_tensor)
                value = self.value(state_tensor)

            # Sample action from the distribution
            m = Categorical(action_probs)
            action = m.sample()

            next_state, reward, done, _ = self.env.step(action.item())

            # Track state visitation for VINE
            state_key = self._get_state_key(state)
            if state_key in self.state_visitation_count:
                self.state_visitation_count[state_key] += 1
            else:
                self.state_visitation_count[state_key] = 1

            # Save data
            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(m.log_prob(action).item())
            dones.append(done)
            found_transformations = done and reward == 0

            state = next_state
            total_reward += reward
            timesteps += 1

        # Convert to tensors
        states_tensor = torch.stack(states)
        actions_tensor = torch.tensor(actions, dtype=torch.float16, device=device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
        values_tensor = torch.tensor(values, dtype=torch.float32, device=device)
        log_probs_tensor = torch.tensor(log_probs, dtype=torch.float32, device=device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool, device=device)

        return {
            'states': states_tensor,
            'actions': actions_tensor,
            'rewards': rewards_tensor,
            'values': values_tensor,
            'log_probs': log_probs_tensor,
            'dones': dones_tensor,
            'total_reward': total_reward,
            'timesteps': timesteps,
            "found_transformations": found_transformations
        }

    def compute_gae(self, trajectory):
        rewards = trajectory['rewards']
        values = trajectory['values']
        dones = trajectory['dones']

        # Apply VINE bonus to rewards
        states = trajectory['states']
        modified_rewards = rewards.clone()

        for i, state in enumerate(states):
            vine_bonus = self.compute_vine_bonus(state)
            modified_rewards[i] += self.vine_bonus_coeff * vine_bonus

        # Calculate the advantages using GAE
        advantages = torch.zeros_like(modified_rewards, device=device)
        last_gae = 0

        # Get the value of the last state if not done
        if not dones[-1]:
            last_state = states[-1]
            with torch.no_grad():
                last_value = self.value(last_state.unsqueeze(0).to(device)).item()
        else:
            last_value = 0

        # Calculate advantages
        for t in reversed(range(len(modified_rewards))):
            if t == len(modified_rewards) - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            delta = modified_rewards[t] + self.gamma * next_value * (1 - dones[t].float()) - values[t]
            last_gae = delta + self.gamma * self.lam * (1 - dones[t].float()) * last_gae
            advantages[t] = last_gae

        # Compute returns for value function training - based on original rewards
        returns = torch.zeros_like(rewards, device=device)
        last_return = last_value
        for t in reversed(range(len(rewards))):
            returns[t] = rewards[t] + self.gamma * last_return * (1 - dones[t].float())
            last_return = returns[t]

        return advantages, returns

    def generate_vine_rollouts(self, states):
        # Generate exploratory rollouts for VINE
        # Sample batch of states to explore from
        if len(states) > self.vine_batch_size:
            indices = torch.randperm(len(states), device=device)[:self.vine_batch_size]
            batch_states = states[indices]
        else:
            batch_states = states

        vine_trajectories = []
        for state in batch_states:
            # Start from this state and apply a random policy for a few steps
            current_state = state.clone() #.cpu().numpy()  # Convert to numpy for env
            self.env.get_env_from_state(current_state)

            vine_states = [current_state]
            vine_actions = []
            vine_rewards = []

            for _ in range(5):  # Short rollout
                # Choose random action
                action = torch.randint(0, self.env.get_action_space(), (1,), device=device).item()
                next_state, reward, done, _ = self.env.step(action)

                vine_states.append(next_state)
                vine_actions.append(action)
                vine_rewards.append(reward)

                if done:
                    break

            # Convert lists to tensors
            vine_states_tensor = torch.stack(vine_states)
            vine_actions_tensor = torch.tensor(vine_actions, dtype=torch.float16, device=device)
            vine_rewards_tensor = torch.tensor(vine_rewards, dtype=torch.float32, device=device)

            vine_trajectories.append({
                'states': vine_states_tensor,
                'actions': vine_actions_tensor,
                'rewards': vine_rewards_tensor
            })

        return vine_trajectories

    def update_vine_value(self, vine_trajectories):
        # Use VINE rollouts to update value function
        all_states = torch.empty(dtype=torch.float16, device=device)
        all_returns = torch.empty(dtype=torch.long, device=device)

        for traj in vine_trajectories:
            states = traj['states']
            rewards = traj['rewards']

            # Calculate returns for this trajectory
            returns = torch.zeros_like(rewards, device=device)
            R = 0
            for i in reversed(range(len(rewards))):
                R = rewards[i] + self.gamma * R
                returns[i] = R

            # Add state-return pairs (excluding the last state)
            if len(states) > 1:
                all_states = torch.cat((all_states,states[:-1]))
                all_returns = torch.cat((all_returns,returns[:-1]))

        if not all_states:  # No data to train on
            return

        # Update value function
        for _ in range(self.vine_epochs):
            # Shuffle data
            indices = torch.randperm(len(all_states), device=device)
            batches = torch.split(indices, self.vine_batch_size)

            for batch_indices in batches:
                if len(batch_indices) < 2:  # Skip tiny batches
                    continue

                batch_states = all_states[batch_indices]
                batch_returns = all_returns[batch_indices].unsqueeze(1)

                value_pred = self.value(batch_states)
                value_loss = F.mse_loss(value_pred, batch_returns)

                self.value_optimizer.zero_grad()
                value_loss.backward()
                self.value_optimizer.step()

    def hessian_vector_product(self, states, old_action_probs, p):
        # Compute Hessian-vector product efficiently using FvP
        def get_kl():
            current_action_probs = self.policy(states)
            kl = F.kl_div(current_action_probs.log(), old_action_probs, reduction='batchmean')
            return kl

        kl = get_kl()
        grads = torch.autograd.grad(kl, self.policy.parameters(), create_graph=True)
        flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

        # Compute product of Hessian with vector p
        kl_p = (flat_grad_kl * p).sum()
        grads_p = torch.autograd.grad(kl_p, self.policy.parameters())
        flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads_p])

        return flat_grad_grad_kl + p * self.cg_damping

    def conjugate_gradient(self, states, old_action_probs, b, nsteps=10, residual_tol=1e-10):
        # Compute x = H^-1 * b using conjugate gradient method
        x = torch.zeros_like(b, device=device)
        r = b.clone()
        p = b.clone()
        rdotr = torch.dot(r, r)

        for i in range(nsteps):
            Hp = self.hessian_vector_product(states, old_action_probs, p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < residual_tol:
                break
            beta = new_rdotr / rdotr
            p = r + beta * p
            rdotr = new_rdotr

        return x

    def compute_policy_gradient(self, states, actions, advantages):
        """Compute the policy gradient"""
        # Get action probabilities
        action_probs = self.policy(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()

        # Compute surrogate loss
        policy_loss = -(log_probs * advantages).mean()

        # Compute gradient
        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        # Get flat gradient
        grad = torch.cat([param.grad.view(-1) for param in self.policy.parameters()]).to(device)
        return grad

    def update_policy(self, trajectory, advantages):
        states = trajectory['states']
        actions = trajectory['actions']

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get current parameters
        old_params = self.get_flat_params()

        # Compute policy gradient
        policy_gradient = self.compute_policy_gradient(states, actions, advantages)

        # Compute step direction and magnitude
        step = policy_gradient
        step_norm = torch.norm(step)
        if step_norm > 0:
            step = step / step_norm

        # Line search in parameter space
        for stepsize in [self.max_param_change * (self.backtrack_coeff ** i)
                         for i in range(self.backtrack_iters)]:

            # Compute proposed new parameters
            new_params = old_params + stepsize * step

            # Enforce trust region constraint
            param_diff = new_params - old_params
            param_norm = torch.norm(param_diff)
            if param_norm > self.max_param_change:
                scaling = self.max_param_change / param_norm
                new_params = old_params + param_diff * scaling

            # Try new parameters
            self.set_flat_params(new_params)

            # Evaluate improvement
            new_loss = self.evaluate_policy(states, actions, advantages)

            # Accept if loss improved
            if new_loss < trajectory.get('old_loss', float('inf')):
                trajectory['old_loss'] = new_loss
                return True

            # Revert parameters if no improvement
            self.set_flat_params(old_params)

        print("Line search failed!")
        return False

    def evaluate_policy(self, states, actions, advantages):
        """Evaluate the policy loss"""
        action_probs = self.policy(states)
        log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1))).squeeze()
        loss = -(log_probs * advantages).mean()

        return loss.item()

    def update_value(self, trajectory, returns):
        states = trajectory['states']
        returns = returns.unsqueeze(1)

        for _ in range(10):  # Multiple updates to value function
            value_pred = self.value(states)
            value_loss = F.mse_loss(value_pred, returns)

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

    def train(self, num_episodes=1000):
        rewards_history = []
        success_history = []
        found_transformations = 0

        # Initialize running statistics
        running_reward = None
        running_loss = None

        for episode in range(num_episodes):
            # Collect trajectory
            trajectory = self.collect_trajectory()

            # Generate VINE rollouts for exploration
            vine_trajectories = self.generate_vine_rollouts(trajectory['states'])

            # Update value function with VINE data
            self.update_vine_value(vine_trajectories)

            # Compute GAE with VINE bonus
            advantages, returns = self.compute_gae(trajectory)

            # Update policy
            success = self.update_policy(trajectory, advantages)
            success_history.append(success)

            # Update value function
            self.update_value(trajectory, returns)

            # Record reward
            episode_reward = trajectory['total_reward']
            rewards_history.append(episode_reward)
            found_transformations += int(trajectory['found_transformations'])

            # Decay VINE bonus coefficient
            self.vine_bonus_coeff *= 0.995  # Gradually reduce exploration bonus

            # Update running statistics
            if running_reward is None:
                running_reward = episode_reward
                running_loss = trajectory.get('old_loss', 0)
            else:
                running_reward = 0.95 * running_reward + 0.05 * episode_reward
                running_loss = 0.95 * running_loss + 0.05 * trajectory.get('old_loss', running_loss)

            # Print progress
            if (episode + 1) % 10 == 0:
                recent_success = torch.tensor(success_history[-10:], dtype=torch.float, device=device)
                success_rate = recent_success.mean().item()
                print(f"Episode {episode + 1}")
                print(f"Running reward: {running_reward:.2f}")
                print(f"Episode reward: {episode_reward:.2f}")
                print(f"Running loss: {running_loss:.4f}")
                print(f"Success rate: {success_rate:.2f}")
                print(f"found transformations in: {found_transformations}")
                found_transformations = 0
                print("------------------------")

        return rewards_history


# Main function to set up and run the training
def main():
    # Set up environment
    env = BraidEnvironment(n_braids_max=10, n_letters_max=20, max_steps=50)

    # Get dimensions
    d_model = env.get_model_dim()  # Dimension of the model
    action_dim = env.get_action_space()
    num_heads = 5  # Number of attention heads

    # Create attention-based networks
    policy_net = BraidAttentionPolicy(d_model=d_model, num_heads=num_heads, output_dim=action_dim)
    value_net = BraidAttentionValue(d_model=d_model, num_heads=num_heads)

    # Create TRPO agent with parameter space trust region
    agent = TRPOAgent(
        env=env,
        policy_network=policy_net,
        value_network=value_net,
        gamma=0.99,
        lam=0.95,
        max_param_change=0.001,  # Maximum allowed parameter change magnitude
        backtrack_iters=10,
        backtrack_coeff=0.8,
        vine_bonus_coeff=0.01,
        vine_batch_size=64,
        vine_epochs=5
    )

    # Train agent
    rewards = agent.train(num_episodes=500)

    # Plot results
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('TRPO with Attention Networks for Braid Transformations')
    plt.savefig('trpo_braid_attention.png')
    plt.show()


if __name__ == "__main__":
    main()
