# MDP Value Iteration and Policy Iteration

import numpy as np
from riverswim import RiverSwim

np.set_printoptions(precision=3)


def bellman_backup(state, action, R, T, gamma, V):
    """
    Perform a single Bellman backup.
    Q(s,a) = sum_{s'} T[s,a,s'] * ( R[s,a] + gamma * V[s'] )
    """
    return np.sum(T[state, action] * (R[state, action] + gamma * V))


def policy_evaluation(policy, R, T, gamma, tol=1e-3):
    """
    Compute the value function induced by a given policy for the input MDP.
    Uses iterative evaluation until convergence.
    """
    num_states, _ = R.shape
    value_function = np.zeros(num_states)

    while True:
        delta = 0.0
        new_value_function = np.zeros_like(value_function)

        for s in range(num_states):
            a = policy[s]
            new_value_function[s] = bellman_backup(
                s, a, R, T, gamma, value_function)
            delta = max(delta, abs(new_value_function[s] - value_function[s]))

        value_function = new_value_function

        if delta < tol:
            break

    return value_function


def policy_improvement(policy, R, T, V_policy, gamma):
    """
    Given V_pi, compute a greedy policy improvement.
    """
    num_states, num_actions = R.shape
    new_policy = np.zeros(num_states, dtype=int)

    for s in range(num_states):
        q_values = []
        for a in range(num_actions):
            q_values.append(bellman_backup(s, a, R, T, gamma, V_policy))
        new_policy[s] = int(np.argmax(q_values))

    return new_policy


def policy_iteration(R, T, gamma, tol=1e-3):
    """
    Runs policy iteration:
      1) policy evaluation
      2) policy improvement
    until the policy is stable.
    """
    num_states, _ = R.shape
    policy = np.zeros(num_states, dtype=int)   # start with all LEFT
    V_policy = np.zeros(num_states)

    while True:
        V_policy = policy_evaluation(policy, R, T, gamma, tol=tol)
        new_policy = policy_improvement(policy, R, T, V_policy, gamma)

        if np.array_equal(new_policy, policy):
            break

        policy = new_policy

    return V_policy, policy


def value_iteration(R, T, gamma, tol=1e-3):
    """
    Runs standard value iteration and extracts the greedy policy.
    """
    num_states, num_actions = R.shape
    value_function = np.zeros(num_states)

    # VALUE UPDATE LOOP
    while True:
        delta = 0.0
        new_value_function = np.zeros_like(value_function)

        for s in range(num_states):
            q_values = []
            for a in range(num_actions):
                q_values.append(bellman_backup(
                    s, a, R, T, gamma, value_function))

            new_value_function[s] = max(q_values)
            delta = max(delta, abs(new_value_function[s] - value_function[s]))

        value_function = new_value_function

        if delta < tol:
            break

    # POLICY EXTRACTION
    policy = np.zeros(num_states, dtype=int)
    num_states, num_actions = R.shape
    for s in range(num_states):
        q_values = []
        for a in range(num_actions):
            q_values.append(bellman_backup(s, a, R, T, gamma, value_function))
        policy[s] = int(np.argmax(q_values))

    return value_function, policy


def find_largest_gamma_with_left_at_start(current, tol=1e-3):
    """
    For a given current strength (WEAK/MEDIUM/STRONG), scan gamma in [0.01, 0.99]
    in steps of 0.01 and return the largest gamma such that the optimal action
    at the leftmost state (state 0) is LEFT.
    """
    env = RiverSwim(current, seed=1234)
    R, T = env.get_model()

    last_gamma_left = None
    for g in np.arange(0.01, 1.0, 0.01):
        g = float(f"{g:.2f}")  # keep two decimals cleanly
        V_vi, policy_vi = value_iteration(R, T, gamma=g, tol=tol)
        # action 0 = LEFT, action 1 = RIGHT
        if policy_vi[0] == 0:
            last_gamma_left = g

    return last_gamma_left


# MAIN: demo runs for Problem 4
if __name__ == "__main__":
    SEED = 1234
    RIVER_CURRENT = "WEAK"   # "WEAK", "MEDIUM", or "STRONG"
    discount_factor = 0.5

    env = RiverSwim(RIVER_CURRENT, SEED)
    R, T = env.get_model()

    print("\n" + "-" * 25 + "\nBeginning Policy Iteration\n" + "-" * 25)
    V_pi, policy_pi = policy_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_pi)
    print([["L", "R"][a] for a in policy_pi])

    print("\n" + "-" * 25 + "\nBeginning Value Iteration\n" + "-" * 25)
    V_vi, policy_vi = value_iteration(R, T, gamma=discount_factor, tol=1e-3)
    print(V_vi)
    print([["L", "R"][a] for a in policy_vi])

    # sweep over gamma for each current strength (for part 4d)
    print("\n" + "-" * 25 + "\nGamma thresholds (Problem 4d)\n" + "-" * 25)
    for current in ["WEAK", "MEDIUM", "STRONG"]:
        g_star = find_largest_gamma_with_left_at_start(current, tol=1e-3)
        print(f"{current}: largest gamma with LEFT optimal at s0 = {g_star:.2f}")
