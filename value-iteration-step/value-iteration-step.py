import numpy as np
def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    values = np.asarray(values, dtype=float)
    transitions = np.asarray(transitions, dtype=float)
    rewards = np.asarray(rewards, dtype=float)

    expected_future = np.sum(transitions * values[None, None, :], axis=2)
    Q_values = rewards + gamma * expected_future

    res_values = list(np.max(Q_values, axis=1))
    return res_values