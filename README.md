# **Reinforcement Learning - Homework Solutions**

This repository contains implementations of foundational **Reinforcement Learning (RL)** algorithms applied to various environments using the **Gymnasium API**. The project explores **Dynamic Programming**, **Monte Carlo methods**, and **Temporal Difference (TD) learning**, comparing their performance, stability, and learned policies across both standard and custom environments.

The repository is divided into two main directories: **HW1** and **HW2**.

## **Directory Structure**
- `HW1/` – Contains the solution for **Homework 1**.
  - `riverswim.py`: Implements the **RiverSwim problem** using **Value Iteration** and **Policy Iteration**.
  - `vi_and_pi.py`: Implements **Value Iteration** and **Policy Iteration** algorithms for the **RiverSwim problem**.

- `HW2/` – Contains the solution for **Homework 2**.
  - `RL_HW2.ipynb`: This Jupyter notebook contains a collection of RL algorithms applied to multiple environments, including **Dynamic Programming** (FrozenLake), **Monte Carlo Control** (Blackjack), and **Temporal Difference Control** (CliffWalking). The notebook includes detailed experiments and analysis on each algorithm's behavior, performance, and learned policies.

---

## **HW1: RiverSwim Problem**

In **Homework 1**, the task was to solve the **RiverSwim problem** using **Dynamic Programming** algorithms.

### **Contents:**
- `riverswim.py`: Defines the **RiverSwim environment**, where the agent has to swim across a river with different current strengths (Weak, Medium, Strong) using two possible actions: **LEFT** or **RIGHT**.
- `vi_and_pi.py`: Implements **Value Iteration** and **Policy Iteration** algorithms to solve the **RiverSwim problem**.

### **Key Concepts Covered:**
- **Markov Decision Processes (MDP)**.
- **Dynamic Programming** algorithms: **Value Iteration** and **Policy Iteration**.
- **Reward** and **Transition matrices**.
  
---

## **HW2: Reinforcement Learning Algorithms**

In **Homework 2**, we explore multiple foundational **Reinforcement Learning** algorithms applied to a range of environments.

### **Algorithms and Environments Covered:**
1. **Dynamic Programming (FrozenLake)**
   - **Algorithms**: Value Iteration and Policy Iteration.
   - **Environments**: Deterministic and Stochastic **FrozenLake-v1.3** (4x4 and 8x8 maps).
   - **Highlights**: 
     - Compared computational efficiency between **Value Iteration** and **Policy Iteration**.
     - Analyzed how introducing stochasticity (slip probability) changes the optimal policy from shortest-path to risk-averse routing.
     - Experimented with a custom reward schedule to observe extreme risk-averse (suicidal) agent behavior.

2. **Custom Environment Design (Server Queue Management)**
   - **Environment**: A custom **Gymnasium** environment simulating a single server managing three parallel queues.
   - **Dynamics**: Implemented custom state/action spaces (MultiDiscrete and Discrete) and probabilistic arrival dynamics for incoming server requests.
   - **Highlights**: Verified environment correctness under symmetric and asymmetric traffic loads.

3. **Monte Carlo Control (Blackjack)**
   - **Algorithms**: 
     - Incremental Soft On-Policy **Monte Carlo Control** (First-Visit).
     - Incremental Soft Off-Policy **Monte Carlo Control** (Weighted Importance Sampling).
   - **Environment**: **Blackjack-v1**.
   - **Highlights**:
     - Explored the **exploration-exploitation** tradeoff using different ϵ values (0.1 and 0.01).
     - Analyzed the variance and stability issues inherent in Off-Policy **Weighted Importance Sampling**.

4. **Temporal Difference Control (Cliff Walking)**
   - **Algorithms**: **SARSA** (On-Policy TD) and **Q-Learning** (Off-Policy TD).
   - **Environment**: **Stochastic CliffWalking-v1**.
   - **Highlights**:
     - Demonstrated the classic **"Safe vs. Optimal"** pathing dilemma.
     - **SARSA** learns a longer, safer route to account for ϵ-greedy exploration risks, while **Q-learning** finds the absolute shortest (but riskiest) path along the cliff edge.
     - Visualized value functions and overlaid optimal policy arrows on a 4x12 grid.

### **Notebook Highlights:**
- Compared the computational efficiency of **Value Iteration** and **Policy Iteration** on the **FrozenLake** problem.
- Analyzed the agent's behavior under different reward structures and stochasticity.
- Implemented a custom environment with complex state and action spaces.
- Investigated the **exploration-exploitation tradeoff** with **Monte Carlo** methods.
- Applied **Temporal Difference learning** algorithms (SARSA and Q-Learning) to solve the **Cliff Walking** problem and compared their performance.

---

## **Installation and Setup**

Before running the notebook, you will need the following libraries:

```bash
pip install gymnasium numpy matplotlib seaborn
