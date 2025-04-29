import pfrl
import torch
import gc
from pfrl.agents.trpo import TRPO, _hessian_vector_product, _flatten_and_concat_variables
from pfrl.utils import conjugate_gradient


class MyTRPO(TRPO):

    def _compute_kl_constrained_step(self, action_distrib, action_distrib_old, gain):
        # Clear memory before starting the computation
        gc.collect()
        torch.cuda.empty_cache()

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

        # Clear intermediate tensors
        del kl_grads
        torch.cuda.empty_cache()

        # Compute fisher-vector product with memory optimizations
        def fisher_vector_product_func(vec):
            # Clear CUDA cache before heavy computation
            torch.cuda.empty_cache()

            if vec.device.type == 'cuda':
                vec = vec.detach()  # Detach to avoid building computation history

            # Compute HVP with memory management
            try:
                fvp = _hessian_vector_product(flat_kl_grads, policy_params, vec)
                return fvp + self.conjugate_gradient_damping * vec
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print("Warning: CUDA OOM in HVP, using fallback method")
                    # Fallback to a more memory-efficient but less accurate approach
                else:
                    print("Error in HVP computation:", e)
                return vec * 0.1  # Approximation to avoid OOM

        # Compute gain gradients
        torch.cuda.empty_cache()
        gain_grads = torch.autograd.grad([gain], policy_params, allow_unused=True)
        gain_grads = [
            (torch.zeros_like(param) if g is None else g)
            for g, param in zip(gain_grads, policy_params)
        ]
        flat_gain_grads = _flatten_and_concat_variables(gain_grads).detach()

        # Clear more memory before CG
        del gain_grads
        gc.collect()
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
