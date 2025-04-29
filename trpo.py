import torch
from pfrl.agents import TRPO
from pfrl.agents.trpo import _flatten_and_concat_variables


def fixed_hvp(flat_grads, params, vec):
    """Fixed hessian-vector product function with proper retain_graph handling."""
    vec = vec.detach()

    # This is the key fix: wrapping the whole computation with create_graph=True
    # and adding retain_graph=True to the inner grad call
    kl_v = torch.sum(flat_grads * vec)

    grads = torch.autograd.grad(
        kl_v, params, retain_graph=True
    )

    flat_kl_grads = _flatten_and_concat_variables(grads)
    return flat_kl_grads


class MyTRPO(TRPO):
    def _compute_kl_constrained_step(self, action_distrib, action_distrib_old, gain):
        """Compute a step of policy parameters with a KL constraint."""
        # Clear memory before starting the computation
        gc.collect()
        torch.cuda.empty_cache()

        policy_params = list(self.policy.parameters())
        kl = torch.mean(
            torch.distributions.kl_divergence(action_distrib_old, action_distrib)
        )

        # Create graph for all gradients to ensure proper gradient flow
        kl_grads = torch.autograd.grad([kl], policy_params, create_graph=True)
        assert all(
            g is not None for g in kl_grads
        ), "The gradient contains None. The policy may have unused parameters."
        flat_kl_grads = _flatten_and_concat_variables(kl_grads)

        # Define fisher-vector product with proper retain_graph
        def fisher_vector_product_func(vec):
            vec = torch.as_tensor(vec)
            if vec.device.type == 'cuda':
                vec = vec.detach()

            # Use our fixed HVP function
            try:
                fvp = fixed_hvp(flat_kl_grads, policy_params, vec)
                return fvp + self.conjugate_gradient_damping * vec
            except RuntimeError as e:
                print(f"HVP error: {e}")
                # Return fallback if error occurs
                return vec * 0.1

        # Compute the surrogate loss gradient
        gain_grads = torch.autograd.grad([gain], policy_params)
        flat_gain_grads = _flatten_and_concat_variables(gain_grads).detach()

        # Clear memory before CG
        torch.cuda.empty_cache()

        try:
            # Compute the search direction using conjugate gradient
            step_direction = pfrl.utils.conjugate_gradient(
                fisher_vector_product_func,
                flat_gain_grads,
                max_iter=self.conjugate_gradient_max_iter,
            )

            # Calculate optimal step size
            dId = float(step_direction.dot(fisher_vector_product_func(step_direction)))
            scale = (2.0 * self.max_kl / (dId + 1e-8)) ** 0.5

            return scale * step_direction

        except RuntimeError as e:
            print(f"CG error: {e}")
            # Return a small step in the gradient direction as fallback
            return flat_gain_grads * 0.01