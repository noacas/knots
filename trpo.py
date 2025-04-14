import pfrl
import torch
from pfrl.agents.trpo import TRPO, _hessian_vector_product, _flatten_and_concat_variables
from pfrl.utils import conjugate_gradient


class MyTRPO(TRPO):
    def _compute_kl_constrained_step(self, action_distrib, action_distrib_old, gain):
        policy_params = list(self.policy.parameters())

        kl = torch.distributions.kl.kl_divergence(
            action_distrib_old, action_distrib
        ).mean()

        # Get kl gradients
        kl_grads = torch.autograd.grad([kl], policy_params, create_graph=True, allow_unused=True)

        # Filter out None gradients by replacing them with zero tensors
        kl_grads = [
            (torch.zeros_like(param) if g is None else g)
            for g, param in zip(kl_grads, policy_params)
        ]
        flat_kl_grads = _flatten_and_concat_variables(kl_grads)
        assert all(g.requires_grad for g in kl_grads)
        assert flat_kl_grads.requires_grad

        def fisher_vector_product_func(vec):
            vec = torch.as_tensor(vec)
            fvp = _hessian_vector_product(flat_kl_grads, policy_params, vec)
            return fvp + self.conjugate_gradient_damping * vec

        gain_grads = torch.autograd.grad([gain], policy_params, create_graph=True)
        assert all(
            g is not None for g in gain_grads
        ), "The gradient contains None. The policy may have unused parameters."
        flat_gain_grads = _flatten_and_concat_variables(gain_grads).detach()
        step_direction = pfrl.utils.conjugate_gradient(
            fisher_vector_product_func,
            flat_gain_grads,
            max_iter=self.conjugate_gradient_max_iter,
        )
        dId = float(step_direction.dot(fisher_vector_product_func(step_direction)))
        scale = (2.0 * self.max_kl / (dId + 1e-8)) ** 0.5
        return scale * step_direction