from pfrl.agents import TRPO


class MyTRPO(TRPO):
    def _compute_gain(self, action_values, actions):
        # Compute probability ratio for importance sampling
        distribs = self.get_policy_distribs_for_probs(action_values)
        distribs_old = self.get_policy_distribs_for_probs(action_values)
        for i in range(len(distribs_old)):
            distribs_old[i].detach_prob_distribution()

        log_probs = []
        log_probs_old = []
        for i in range(len(distribs)):
            log_probs.append(distribs[i].log_prob(actions[i]))
            log_probs_old.append(distribs_old[i].log_prob(actions[i]))

        log_probs = torch.stack(log_probs)
        log_probs_old = torch.stack(log_probs_old)

        # Advantage: compute on CPU for stability
        with torch.no_grad():
            vs = self.get_batch_value_prediction(action_values)
            next_vs = self.get_batch_value_prediction(
                [elem["next_state"] for elem in action_values]
            )
            advs = [
                elem["reward"]
                + (
                    0
                    if elem["is_state_terminal"]
                    else self.gamma * next_vs[i].item()
                )
                - vs[i].item()
                for i, elem in enumerate(action_values)
            ]
            advs = torch.tensor(advs, dtype=torch.float)

        # Normalize advantage
        n_samples = advs.shape[0]
        mean = torch.mean(advs)
        advs = advs - mean
        std = torch.std(advs)
        if std != 0:
            advs = advs / std
        else:
            advs = advs * 0

        prob_ratio = torch.exp(log_probs - log_probs_old)
        gain = torch.sum(prob_ratio * advs) / n_samples
        return gain, distribs, distribs_old

    def update(self, experiences, errors_out=None):
        """Update the model using a batch of experiences

        Args:
            experiences: List of experiences
            errors_out (list or None): Optional output destination for errors

        Returns:
            None
        """
        # Generate parser for experiences
        action_values = list(filter(lambda elem: "state" in elem, experiences))

        # Compute loss
        gain, distribs, distribs_old = self._compute_gain(action_values,
                                                          [elem["action"] for elem in action_values])

        # Update policy with kl constraint
        step_size = self._compute_kl_constrained_step(distribs, distribs_old, gain)

        # Apply update
        self.policy_optimizer.zero_grad()
        flat_params = _flatten_and_concat_variables(list(self.policy.parameters()))
        updated_flat_params = flat_params + step_size
        _replace_params_data(self.policy, updated_flat_params)

        # Update value function
        vsgen = (elem["state"] for elem in experiences)
        vs = self.get_batch_value_prediction(list(vsgen))
        target_vs = []
        target_vs_in_next_step = self.get_batch_value_prediction(
            [elem["next_state"] for elem in experiences]
        )

        for i, (exp, target_v_in_next_step) in enumerate(
                zip(experiences, target_vs_in_next_step)
        ):
            if exp["is_state_terminal"]:
                target_vs.append(torch.tensor(exp["reward"]))
            else:
                target_vs.append(
                    torch.tensor(
                        exp["reward"] + self.gamma * target_v_in_next_step.item()
                    )
                )

        target_vs = torch.stack(target_vs)
        for _ in range(10):  # Multiple value updates per policy update
            self.value_function_optimizer.zero_grad()
            loss = ((vs - target_vs) ** 2).mean() / 2
            loss.backward()
            self.value_function_optimizer.step()
            vs = self.get_batch_value_prediction(list(vsgen))  # Refresh vs

        return None


# Helper function for updating parameters
def _replace_params_data(module, flat_params):
    """Replace data of params in module with given flat parameters.

    Args:
        module (torch.nn.Module): Module with parameters to be replaced
        flat_params (torch.Tensor): Flattened parameter data
    """
    offset = 0
    for param in module.parameters():
        new_data = flat_params[offset: offset + param.numel()].reshape(param.shape)
        param.data.copy_(new_data)
        offset += param.numel()