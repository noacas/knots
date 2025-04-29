import pfrl
import torch
import gc
import numpy as np
from pfrl.agents.trpo import TRPO, _flatten_and_concat_variables
from pfrl.utils import conjugate_gradient
from pfrl.utils.batch_states import batch_states
import torch.nn.functional as F
from pfrl.utils import clip_l2_grad_norm_
from pfrl.agents.ppo import _yield_minibatches


class FixedTRPO(TRPO):
    def _compute_kl_constrained_step(self, action_distrib, action_distrib_old, gain):
        """Compute a step of policy parameters with a KL constraint."""
        # Clear memory before starting the computation
        gc.collect()
        torch.cuda.empty_cache()

        policy_params = list(self.policy.parameters())

        kl = torch.mean(
            torch.distributions.kl_divergence(action_distrib_old, action_distrib)
        )

        # Get kl gradients - ADD allow_unused=True HERE
        kl_grads = torch.autograd.grad([kl], policy_params, create_graph=True, allow_unused=True)

        # Filter out None gradients by replacing them with zero tensors
        kl_grads = [
            (torch.zeros_like(param) if g is None else g)
            for g, param in zip(kl_grads, policy_params)
        ]
        flat_kl_grads = _flatten_and_concat_variables(kl_grads)

        # Clear intermediate tensors
        del kl_grads
        torch.cuda.empty_cache()

        # Compute fisher-vector product with memory optimizations
        def fisher_vector_product_func(vec):
            # Clear CUDA cache before heavy computation
            torch.cuda.empty_cache()

            if vec.device.type == 'cuda':
                vec = vec.detach()  # Detach to avoid building computation history

            # Compute HVP with fixed version that includes retain_graph=True
            try:
                vec = vec.detach()
                kl_v = (flat_kl_grads * vec).sum()

                # Add allow_unused=True HERE TOO
                grads = torch.autograd.grad(kl_v, policy_params, retain_graph=True, allow_unused=True)

                # Replace None gradients with zeros
                grads = [
                    (torch.zeros_like(param) if g is None else g)
                    for g, param in zip(grads, policy_params)
                ]

                flat_kl_grads2 = _flatten_and_concat_variables(grads)
                return flat_kl_grads2 + self.conjugate_gradient_damping * vec

            except RuntimeError as e:
                print("Error in HVP computation:", e)
                return vec * 0.1  # Approximation to avoid OOM

        # Compute gain gradients - ADD allow_unused=True HERE TOO
        torch.cuda.empty_cache()
        gain_grads = torch.autograd.grad([gain], policy_params, allow_unused=True)
        gain_grads = [
            (torch.zeros_like(param) if g is None else g)
            for g, param in zip(gain_grads, policy_params)
        ]
        flat_gain_grads = _flatten_and_concat_variables(gain_grads).detach()

        # Clear more memory before CG
        del gain_grads
        torch.cuda.empty_cache()

        try:
            step_direction = pfrl.utils.conjugate_gradient(
                fisher_vector_product_func,
                flat_gain_grads,
                max_iter=self.conjugate_gradient_max_iter,
            )

            # Calculate the scale factor
            dId = float(step_direction.dot(fisher_vector_product_func(step_direction)))
            scale = (2.0 * self.max_kl / (dId + 1e-8)) ** 0.5

            # Clear memory after computation
            gc.collect()
            torch.cuda.empty_cache()

            return scale * step_direction

        except RuntimeError as e:
            print(f"Error in conjugate gradient: {e}")
            # Fallback if CG fails
            print("Using gradient direction as fallback")
            torch.cuda.empty_cache()
            # Just use the gradient direction with a small step size
            return flat_gain_grads * 0.01


class EnhancedTRPO(FixedTRPO):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Increase number of epochs for value function training
        self.vf_epochs = 10

        # Set a higher learning rate for value function
        for param_group in self.vf_optimizer.param_groups:
            param_group['lr'] *= 3.0

    def _update_vf(self, dataset):
        """Enhanced value function update with better training"""
        assert "state" in dataset[0]
        assert "v_teacher" in dataset[0]

        # Get all returns for diagnostics
        all_returns = np.array([b["v_teacher"] for b in dataset])

        print(f"Returns stats: min={all_returns.min():.4f}, "
              f"max={all_returns.max():.4f}, "
              f"mean={all_returns.mean():.4f}, "
              f"std={all_returns.std():.4f}")

        # Multiple epochs of training
        for epoch in range(self.vf_epochs):
            for batch in _yield_minibatches(
                    dataset, minibatch_size=self.vf_batch_size, num_epochs=1
            ):
                states = batch_states([b["state"] for b in batch], self.device, self.phi)
                if self.obs_normalizer:
                    states = self.obs_normalizer(states, update=False)

                vs_teacher = torch.as_tensor(
                    [b["v_teacher"] for b in batch],
                    device=self.device,
                    dtype=torch.float,
                )

                vs_pred = self.vf(states).squeeze(-1)
                vf_loss = F.mse_loss(vs_pred, vs_teacher)

                self.vf_optimizer.zero_grad()
                vf_loss.backward()
                if self.max_grad_norm is not None:
                    clip_l2_grad_norm_(self.vf.parameters(), self.max_grad_norm)
                self.vf_optimizer.step()

        # Calculate explained variance
        with torch.no_grad():
            all_states = batch_states([b["state"] for b in dataset], self.device, self.phi)
            if self.obs_normalizer:
                all_states = self.obs_normalizer(all_states, update=False)

            predictions = self.vf(all_states).squeeze(-1).cpu().numpy()
            targets = all_returns
            var_y = np.var(targets)
            explained_var = 1 - np.var(targets - predictions) / (var_y + 1e-8)

            print(f"Value function explained variance: {explained_var:.4f}")

            # Update our saved explained variance
            self.explained_variance = explained_var