import pfrl
import torch
import gc
import numpy as np
from pfrl.agents.trpo import TRPO, _flatten_and_concat_variables
from pfrl.utils import conjugate_gradient


def safe_hvp(flat_kl_grads, policy_params, vec):
    """
    Safe Hessian-vector product computation that avoids graph retention issues.

    This works by using finite differences approximation as a fallback.
    """
    # First try the direct approach with retain_graph=True
    try:
        vec = vec.detach()
        kl_v = (flat_kl_grads * vec).sum()

        grads = torch.autograd.grad(
            kl_v, policy_params, retain_graph=True, allow_unused=True
        )

        # Replace None gradients with zeros
        grads = [
            torch.zeros_like(param) if g is None else g
            for g, param in zip(grads, policy_params)
        ]

        return _flatten_and_concat_variables(grads)

    except RuntimeError as e:
        print(f"Direct HVP failed with error: {e}")
        print("Using finite difference approximation instead")

        # Use finite difference approximation
        # Save original parameters
        original_params = [p.data.clone() for p in policy_params]

        # Compute parameter shapes and sizes for later
        param_shapes = [p.shape for p in policy_params]
        param_sizes = [p.numel() for p in policy_params]

        # Flatten vector to make it easier to work with
        flat_vec = vec.clone().detach()

        # Choose a small epsilon
        eps = 1e-3

        # Perturb parameters in the direction of the vector
        with torch.no_grad():
            offset = 0
            for i, param in enumerate(policy_params):
                size = param_sizes[i]
                shape = param_shapes[i]
                # Extract portion of the vector corresponding to this parameter
                param_vec = flat_vec[offset:offset + size].reshape(shape)
                # Perturb parameter
                param.data.add_(eps * param_vec)
                offset += size

        # Compute perturbed KL gradients
        with torch.enable_grad():
            # Forward pass to get new distribution
            states = [b["state"] for b in dataset]
            states_tensor = torch.tensor(states, device=policy_params[0].device)
            perturbed_distrib = policy(states_tensor)

            # Compute KL divergence
            with torch.no_grad():
                old_distrib = torch.distributions.Categorical(probs=old_probs)

            kl = torch.mean(torch.distributions.kl_divergence(
                old_distrib, perturbed_distrib
            ))

            # Get gradients of KL
            perturbed_grads = torch.autograd.grad(
                kl, policy_params, allow_unused=True
            )

            # Replace None with zeros
            perturbed_grads = [
                torch.zeros_like(param) if g is None else g.detach()
                for g, param in zip(perturbed_grads, policy_params)
            ]

        # Restore original parameters
        with torch.no_grad():
            for i, param in enumerate(policy_params):
                param.data.copy_(original_params[i])

        # Compute finite difference approximation
        fd_grads = []
        for i in range(len(perturbed_grads)):
            diff = (perturbed_grads[i] - kl_grads[i]) / eps
            fd_grads.append(diff)

        return _flatten_and_concat_variables(fd_grads)


class MyTRPO(TRPO):
    """TRPO with robust KL-constrained updates."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Increase epochs for value function
        self.vf_epochs = 10

    def _compute_kl_constrained_step(self, action_distrib, action_distrib_old, gain):
        """Compute a step with robust HVP calculation."""
        policy_params = list(self.policy.parameters())

        # Store variables for the HVP calculation
        global policy, old_probs, dataset, kl_grads
        policy = self.policy
        old_probs = action_distrib_old.probs.detach()

        # Compute KL divergence
        kl = torch.mean(
            torch.distributions.kl_divergence(action_distrib_old, action_distrib)
        )

        # Get KL gradients
        kl_grads = torch.autograd.grad(
            [kl], policy_params, create_graph=False, allow_unused=True
        )

        # Replace None gradients with zeros
        kl_grads = [
            torch.zeros_like(param) if g is None else g
            for g, param in zip(kl_grads, policy_params)
        ]

        flat_kl_grads = _flatten_and_concat_variables(kl_grads)

        # Define FVP function using our safe HVP
        def fisher_vector_product_func(vec):
            return safe_hvp(flat_kl_grads, policy_params, vec) + self.conjugate_gradient_damping * vec

        # Compute gain gradients
        gain_grads = torch.autograd.grad(
            [gain], policy_params, allow_unused=True
        )

        # Replace None gradients with zeros
        gain_grads = [
            torch.zeros_like(param) if g is None else g
            for g, param in zip(gain_grads, policy_params)
        ]

        flat_gain_grads = _flatten_and_concat_variables(gain_grads).detach()

        try:
            # Try conjugate gradient
            step_direction = conjugate_gradient(
                fisher_vector_product_func,
                flat_gain_grads,
                max_iter=self.conjugate_gradient_max_iter,
            )

            # Calculate scaling factor
            hvp_step = fisher_vector_product_func(step_direction)
            scale = torch.sqrt(
                2.0 * self.max_kl / (torch.dot(step_direction, hvp_step) + 1e-8)
            )

            return scale * step_direction

        except RuntimeError as e:
            print(f"Error in conjugate gradient: {e}")
            # Fallback to simple gradient step
            norm = torch.norm(flat_gain_grads)
            if norm > 0:
                direction = flat_gain_grads / norm
            else:
                direction = flat_gain_grads

            # Start with a small step size
            return 0.01 * direction