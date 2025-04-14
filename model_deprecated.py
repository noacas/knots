import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from braid_env import BraidEnvironment


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
        # self.position_encoding = nn.Parameter(torch.randn(1, 100, d_model))  # Max length 100

        # Layer 1
        self.self_attention1 = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attention1 = nn.MultiheadAttention(d_model, num_heads)
        self.ff1_1 = nn.Linear(d_model, d_model * 4)
        self.ff1_2 = nn.Linear(d_model * 4, d_model)
        self.norm1_1 = nn.LayerNorm(d_model)
        self.norm1_2 = nn.LayerNorm(d_model)
        self.norm1_3 = nn.LayerNorm(d_model)

        # Layer 2
        self.self_attention2 = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attention2 = nn.MultiheadAttention(d_model, num_heads)
        self.ff2_1 = nn.Linear(d_model, d_model * 4)
        self.ff2_2 = nn.Linear(d_model * 4, d_model)
        self.norm2_1 = nn.LayerNorm(d_model)
        self.norm2_2 = nn.LayerNorm(d_model)
        self.norm2_3 = nn.LayerNorm(d_model)

        # Layer 3
        self.self_attention3 = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attention3 = nn.MultiheadAttention(d_model, num_heads)
        self.ff3_1 = nn.Linear(d_model, d_model * 4)
        self.ff3_2 = nn.Linear(d_model * 4, d_model)
        self.norm3_1 = nn.LayerNorm(d_model)
        self.norm3_2 = nn.LayerNorm(d_model)
        self.norm3_3 = nn.LayerNorm(d_model)

        # Final refinement layer
        self.final_self_attention = nn.MultiheadAttention(d_model, num_heads)
        self.final_norm = nn.LayerNorm(d_model)

        # Output layers
        self.output_hidden = nn.Linear(d_model, d_model * 2)
        self.output = nn.Linear(d_model * 2, output_dim)

        self.ln1 = nn.LayerNorm(d_model * 2)
        self.ln2 = nn.LayerNorm(output_dim)

        # Dropout
        self.dropout = nn.Dropout(0.1)
        self.output_dropout = nn.Dropout(0.1 * 1.5)  # Slightly higher dropout for final layer

    def forward(self, x):
        # Split input into current and target braids
        batch_size = x.size(0)
        seq_len = x.size(1) // 2
        current_braid = x[:, :seq_len].unsqueeze(-1).to(device)
        target_braid = x[:, seq_len:].unsqueeze(-1).to(device)

        # Embed current and target braids
        current = self.current_embedding(current_braid)
        target = self.target_embedding(target_braid)

        # # Add positional encoding
        # current = current + self.position_encoding[:, :seq_len, :]
        # target = target + self.position_encoding[:, :seq_len, :]

        # ----- Multiple layers of attention -----

        # Layer 1
        # Self-attention on current braid
        self_attn_output1, _ = self.self_attention1(current, current, current)
        current = self.norm1_1(current + self.dropout(self_attn_output1))

        # Cross-attention between current and target braids
        cross_attn_output1, _ = self.cross_attention1(current, target, target)
        current = self.norm1_2(current + self.dropout(cross_attn_output1))

        # Feed-forward for first layer
        ff_output1 = self.ff1_2(F.relu(self.ff1_1(current)))
        current = self.norm1_3(current + self.dropout(ff_output1))

        # Layer 2
        # Self-attention on current braid (second layer)
        self_attn_output2, _ = self.self_attention2(current, current, current)
        current = self.norm2_1(current + self.dropout(self_attn_output2))

        # Cross-attention between current and target braids (second layer)
        cross_attn_output2, _ = self.cross_attention2(current, target, target)
        current = self.norm2_2(current + self.dropout(cross_attn_output2))

        # Feed-forward for second layer
        ff_output2 = self.ff2_2(F.relu(self.ff2_1(current)))
        current = self.norm2_3(current + self.dropout(ff_output2))

        # Layer 3
        # Self-attention on current braid (third layer)
        self_attn_output3, _ = self.self_attention3(current, current, current)
        current = self.norm3_1(current + self.dropout(self_attn_output3))

        # Cross-attention between current and target braids (third layer)
        cross_attn_output3, _ = self.cross_attention3(current, target, target)
        current = self.norm3_2(current + self.dropout(cross_attn_output3))

        # Feed-forward for third layer
        ff_output3 = self.ff3_2(F.relu(self.ff3_1(current)))
        current = self.norm3_3(current + self.dropout(ff_output3))

        # Additional layer of self-attention for final refinement
        final_self_attn, _ = self.final_self_attention(current, current, current)
        current = self.final_norm(current + self.dropout(final_self_attn))

        # Pool across sequence dimension and project to output dimension
        current = current.mean(dim=1)

        # Final projection with deeper networks
        hidden = self.ln1(F.relu(self.output_hidden(current)))
        hidden = self.output_dropout(hidden)
        output = self.ln2(self.output(hidden))

        # Use a safer softmax approach
        output = F.softmax(output.clamp(-20, 20), dim=-1)

        # Extra safeguard against NaN
        output = torch.where(torch.isnan(output), torch.tensor(1e-8, device=output.device), output)

        # Renormalize to ensure a valid probability distribution
        output = output / output.sum(dim=-1, keepdim=True).clamp(min=1e-10)

        return output


# BraidAttentionValue Network
class BraidAttentionValue(nn.Module):
    def __init__(self, d_model, num_heads):
        super(BraidAttentionValue, self).__init__()

        self.d_model = d_model

        # Embedding layers
        self.current_embedding = nn.Linear(1, d_model)
        self.target_embedding = nn.Linear(1, d_model)

        # Positional encoding
        # self.position_encoding = nn.Parameter(torch.randn(1, 100, d_model))  # Max length 100

        # Layer 1
        self.self_attention1 = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attention1 = nn.MultiheadAttention(d_model, num_heads)
        self.ff1_1 = nn.Linear(d_model, d_model * 4)
        self.ff1_2 = nn.Linear(d_model * 4, d_model)
        self.norm1_1 = nn.LayerNorm(d_model)
        self.norm1_2 = nn.LayerNorm(d_model)
        self.norm1_3 = nn.LayerNorm(d_model)

        # Layer 2
        self.self_attention2 = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attention2 = nn.MultiheadAttention(d_model, num_heads)
        self.ff2_1 = nn.Linear(d_model, d_model * 4)
        self.ff2_2 = nn.Linear(d_model * 4, d_model)
        self.norm2_1 = nn.LayerNorm(d_model)
        self.norm2_2 = nn.LayerNorm(d_model)
        self.norm2_3 = nn.LayerNorm(d_model)

        # Layer 3
        self.self_attention3 = nn.MultiheadAttention(d_model, num_heads)
        self.cross_attention3 = nn.MultiheadAttention(d_model, num_heads)
        self.ff3_1 = nn.Linear(d_model, d_model * 4)
        self.ff3_2 = nn.Linear(d_model * 4, d_model)
        self.norm3_1 = nn.LayerNorm(d_model)
        self.norm3_2 = nn.LayerNorm(d_model)
        self.norm3_3 = nn.LayerNorm(d_model)

        # Final attention layer
        self.final_attention = nn.MultiheadAttention(d_model, num_heads)
        self.final_norm = nn.LayerNorm(d_model)

        # Output layer
        self.output = nn.Linear(d_model, 1)

        # Dropout
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        # Split input into current and target braids
        batch_size = x.size(0)
        seq_len = x.size(1) // 2
        current_braid = x[:, :seq_len].unsqueeze(-1)
        target_braid = x[:, seq_len:].unsqueeze(-1)

        # Embed current and target braids
        current = self.current_embedding(current_braid)
        target = self.target_embedding(target_braid)

        # # Add positional encoding
        # current = current + self.position_encoding[:, :seq_len, :]
        # target = target + self.position_encoding[:, :seq_len, :]

        # Layer 1
        # Self-attention on current braid
        self_attn_output1, _ = self.self_attention1(current, current, current)
        current = self.norm1_1(current + self.dropout(self_attn_output1))

        # Cross-attention between current and target braids
        cross_attn_output1, _ = self.cross_attention1(current, target, target)
        current = self.norm1_2(current + self.dropout(cross_attn_output1))

        # Feed-forward for first layer
        ff_output1 = self.ff1_2(F.relu(self.ff1_1(current)))
        current = self.norm1_3(current + self.dropout(ff_output1))

        # Layer 2
        # Self-attention on current braid
        self_attn_output2, _ = self.self_attention2(current, current, current)
        current = self.norm2_1(current + self.dropout(self_attn_output2))

        # Cross-attention between current and target braids
        cross_attn_output2, _ = self.cross_attention2(current, target, target)
        current = self.norm2_2(current + self.dropout(cross_attn_output2))

        # Feed-forward for second layer
        ff_output2 = self.ff2_2(F.relu(self.ff2_1(current)))
        current = self.norm2_3(current + self.dropout(ff_output2))

        # Layer 3
        # Self-attention on current braid
        self_attn_output3, _ = self.self_attention3(current, current, current)
        current = self.norm3_1(current + self.dropout(self_attn_output3))

        # Cross-attention with target braid
        cross_attn_output3, _ = self.cross_attention3(current, target, target)
        current = self.norm3_2(current + self.dropout(cross_attn_output3))

        # Feed-forward for third layer
        ff_output3 = self.ff3_2(F.relu(self.ff3_1(current)))
        current = self.norm3_3(current + self.dropout(ff_output3))

        # Final attention layer
        final_attn_output, _ = self.final_attention(current, current, current)
        current = self.final_norm(current + self.dropout(final_attn_output))

        # Pool across sequence dimension and project to output dimension
        current = current.mean(dim=1)
        output = self.output(current)

        return output


class TRPOAgent:
    def __init__(self, env, policy_network, value_network, gamma=0.99, lam=0.95,
                 max_param_change=0.01, backtrack_iters=10, backtrack_coeff=0.8,
                 max_timesteps=1000, vine_bonus_coeff=0.01,
                 vine_batch_size=64, vine_epochs=10):
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

        self.update_counter = 0  # Track number of policy updates

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
            return 1.0 / torch.sqrt(torch.tensor(count, dtype=torch.float, device=device))

    def collect_trajectory(self):
        states = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []

        similarities = []

        state = self.env.reset()
        done = False
        total_reward = 0
        timesteps = 0
        found_transformations = False

        while not done:
            state_tensor = state.unsqueeze(0).to(device)
            # Get action probabilities from policy
            with torch.no_grad():
                action_probs = self.policy(state_tensor)
                value = self.value(state_tensor)

            # Sample action from the distribution
            m = Categorical(action_probs)
            action = m.sample()

            #next_state, reward, done, found_transformations, similarity = self.env.step(action.item())

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
            similarities.append(similarity)

            state = next_state
            total_reward += reward
            timesteps += 1

            if found_transformations:
                print(f"success on {self.env.start_braid} to {self.env.target_braid}")
                # for i, move in enumerate(self.env.chosen_moves):
                #     braid = self.env.intermediate_braids[i]
                #     print(f"move: {move}, braid: {braid}")
                print(f"total steps: {self.env.steps_taken}")
                print("----------------------------------")

        # Convert to tensors
        states_tensor = torch.stack(states).to(device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float, device=device)
        values_tensor = torch.tensor(values, dtype=torch.float, device=device)
        log_probs_tensor = torch.tensor(log_probs, dtype=torch.float, device=device)
        dones_tensor = torch.tensor(dones, dtype=torch.bool, device=device)
        similarities_tensor = torch.tensor(similarities, dtype=torch.float, device=device)

        return {
            'states': states_tensor,
            'actions': actions_tensor,
            'rewards': rewards_tensor,
            'values': values_tensor,
            'log_probs': log_probs_tensor,
            'dones': dones_tensor,
            'total_reward': total_reward,
            'timesteps': timesteps,
            "found_transformations": found_transformations,
            "similarities": similarities_tensor,
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
            env = self.env.get_env_from_state(current_state)

            vine_states = [current_state]
            vine_actions = []
            vine_rewards = []

            for _ in range(5): #5?  # Short rollout
                # Choose random action
                action = torch.randint(0, env.get_action_space(), (1,), device=device).item()
                #next_state, reward, done, _, _ = env.step(action)
                next_state = next_state.to(device)

                vine_states.append(next_state)
                vine_actions.append(action)
                vine_rewards.append(reward)

                if done:
                    break

            # Convert lists to tensors
            vine_states_tensor = torch.stack(vine_states)
            vine_actions_tensor = torch.tensor(vine_actions, dtype=torch.float, device=device)
            vine_rewards_tensor = torch.tensor(vine_rewards, dtype=torch.float, device=device)

            vine_trajectories.append({
                'states': vine_states_tensor,
                'actions': vine_actions_tensor,
                'rewards': vine_rewards_tensor
            })

        return vine_trajectories

    def update_vine_value(self, vine_trajectories):
        # Use VINE rollouts to update value function
        all_states = []
        all_returns = []

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
                all_states.append(states[:-1])
                all_returns.append(returns[:-1])

        if not all_states:  # No data to train on
            return

        all_states = torch.cat(all_states, dim=0).to(device)
        all_returns = torch.cat(all_returns, dim=0).to(device)

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
        log_probs = torch.log(action_probs.gather(1, actions.long().unsqueeze(1))).squeeze()

        # Compute surrogate loss
        policy_loss = -(log_probs * advantages).mean()

        # Compute gradient
        self.policy_optimizer.zero_grad()
        policy_loss.backward()

        # Get flat gradient
        grad = torch.cat([param.grad.view(-1) for param in self.policy.parameters()]).to(device)

        # Clip gradient to prevent extreme values
        max_grad_norm = 5.0
        grad_norm = torch.norm(grad)
        if grad_norm > max_grad_norm:
            grad = grad * max_grad_norm / grad_norm

        return grad

    def update_policy(self, trajectory, advantages):
        states = trajectory['states']
        actions = trajectory['actions']
        iteration = getattr(self, 'update_counter', 0)  # Track update iterations

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Get current parameters and action probabilities
        old_params = self.get_flat_params()
        with torch.no_grad():
            old_action_probs = self.policy(states)

        # Compute initial loss for comparison
        old_loss = self.evaluate_policy(states, actions, advantages)

        # Compute policy gradient
        policy_gradient = self.compute_policy_gradient(states, actions, advantages)
        gradient_norm = torch.norm(policy_gradient).item()

        # Set damping coefficient for CG
        self.cg_damping = 0.2

        # Use conjugate gradient to compute Hx = g
        step_direction = self.conjugate_gradient(states, old_action_probs, policy_gradient)

        # Compute step size using Hessian-vector product
        shs = 0.5 * (step_direction * self.hessian_vector_product(states, old_action_probs, step_direction)).sum()
        lm = torch.sqrt(2 * self.max_param_change / (shs + 1e-8))
        full_step = step_direction * lm

        # Line search in parameter space
        success = False
        tiny_step_factor = 1e-4

        for stepsize in [1.0 * (self.backtrack_coeff ** i) for i in range(self.backtrack_iters)]:
            # Compute proposed new parameters
            new_params = old_params + stepsize * full_step

            # Try new parameters
            self.set_flat_params(new_params)

            # Evaluate improvement with extra checks for numerical stability
            try:
                new_loss = self.evaluate_policy(states, actions, advantages)

                # Check if the loss is finite
                if not torch.isfinite(torch.tensor(new_loss)):
                    continue

                # Calculate KL divergence
                with torch.no_grad():
                    current_action_probs = self.policy(states)
                    kl_div = F.kl_div(current_action_probs.log(), old_action_probs, reduction='batchmean')

                # Only accept if improvement is positive and KL is in bounds
                actual_improve = old_loss - new_loss
                if actual_improve > 0 and kl_div < self.max_param_change:
                    trajectory['old_loss'] = new_loss
                    success = True
                    break
            except RuntimeError as e:
                print(f"Runtime error in line search: {e}")
                continue

        return success

    def log_update_info(self, iteration, kl, loss_before, loss_after, gradient_norm):
        print(f"Update {iteration}:")
        print(f"  KL divergence: {kl:.6f}")
        print(f"  Loss before: {loss_before:.6f}, after: {loss_after:.6f}")
        print(f"  Improvement: {loss_before - loss_after:.6f}")
        print(f"  Gradient norm: {gradient_norm:.6f}")

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
        similarities_history = []
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

            # Record similarity
            similarities_history.append(trajectory['similarities'][-1])

            # Decay VINE bonus coefficient
            self.vine_bonus_coeff *= 0.995  # Gradually reduce exploration bonus

            # Update running statistics
            if running_reward is None:
                running_reward = episode_reward
                running_loss = trajectory.get('old_loss', 0)
            else:
                running_reward = 0.95 * running_reward + 0.05 * episode_reward
                running_loss = 0.95 * running_loss + 0.05 * trajectory.get('old_loss', running_loss)

            print("Similarity progress: ", trajectory['similarities'])

            # Print progress
            if (episode + 1) % 10 == 0:
                recent_success = torch.tensor(success_history[-10:], dtype=torch.float, device=device)
                success_rate = recent_success.mean().item()
                print(f"Episode {episode + 1}")
                print(f"Running reward: {running_reward:.2f}")
                print(f"Episode reward: {episode_reward:.2f}")
                print(f"Running loss: {running_loss:.4f}")
                print(f"Success rate: {success_rate:.2f}")
                print(f"Found transformations in: {found_transformations}")
                print(f"Final similarities: {similarities_history}")
                found_transformations = 0
                similarities_history = []
                print("------------------------")
            if episode == 400:
                continue

        return rewards_history


# Main function to set up and run the training
def main():
    # Set up environment
    env = BraidEnvironment(n_braids_max=5, n_letters_max=10, max_steps=10, max_steps_in_generation=1)

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
        max_param_change=0.2,  # Maximum allowed parameter change magnitude
        backtrack_iters=10,
        backtrack_coeff=0.8,
        vine_bonus_coeff=0.3,
        vine_batch_size=64,
        vine_epochs=20
    )

    # Train agent
    rewards = agent.train(num_episodes=600)

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
