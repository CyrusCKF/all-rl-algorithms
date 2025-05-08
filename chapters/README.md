# All RL Algorithms from Scratch ✨

This repository is a collection of Python implementations of various Reinforcement Learning (RL) algorithms. The *primary* goal is **educational**: to get a deep and intuitive understanding of how these algorithms work under the hood. 🧠 Due to the recent explosion in the AI domain especially Large Language Models, and many more applications it is important to understand core reinforcement learning algorithms.

This repository also includes a [comprehensive cheat sheet](cheatsheet.md) summarizing key concepts and algorithms for quick reference.

**This is *not* a performance-optimized library!** I prioritize readability and clarity over speed and advanced features. Think of it as your interactive textbook for RL.

## Updates:
- **2 April 2025**: Added a comprehensive [RL Cheat Sheet](cheatsheet.md) summarizing all implemented algorithms and core concepts. Repository now includes 18 algorithm notebooks.
- **30 March 2025**: Added 18 new algorithms.

## 🌟 Why This Repo?

- **Focus on Fundamentals:** Learn the core logic *without* the abstraction of complex RL libraries. We use basic libraries (NumPy, Matplotlib, PyTorch) to get our hands dirty. 🛠️
- **Beginner-Friendly:** Step-by-step explanations guide you through each algorithm, even if you're new to RL. 👶
- **Interactive Learning:** Jupyter Notebooks provide a playground for experimentation. Tweak hyperparameters, modify the code, and *see* what happens! 🧪
- **Clear and Concise Code:** We strive for readable code that closely mirrors the mathematical descriptions of the algorithms. No unnecessary complexity! 👌
- **Quick Reference:** Includes a detailed [Cheat Sheet](cheatsheet.md) for fast lookups of formulas, pseudocode, and concepts.

## 🗺️ Roadmap: Algorithms Covered (and Coming Soon)

The repository currently includes implementations of the following RL algorithms, with more planned:

**Algorithm Quick Reference**

| Algorithm                                                      | Type          | Description                                                                                                                                                        | Notebook                                              |
| :------------------------------------------------------------- | :------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------------------------------------------------- |
| [Simple Exploration Bot](1_simple_rl.ipynb)                 | Basic         | Demonstrates the core loop: interacting with the environment and storing experienced rewards for later action selection. *Does not actually learn in a true RL sense*. | [1_simple_rl.ipynb](1_simple_rl.ipynb)                 |
| [Q-Learning](2_q_learning.ipynb)                               | Value-Based   | Learns an optimal action-value function (Q-function) through the Bellman equation, enabling goal-directed behavior.                                               | [2_q_learning.ipynb](2_q_learning.ipynb)                 |
| [SARSA](3_sarsa.ipynb)                                          | Value-Based   | On-policy learning algorithm that updates Q-values based on the actions actually taken, often resulting in more cautious behavior.                                   | [3_sarsa.ipynb](3_sarsa.ipynb)                           |
| [Expected SARSA](4_expected_sarsa.ipynb)                      | Value-Based   | On-policy with reduced variance, updates Q-values using the expected value of next actions, balancing exploration and exploitation.                                 | [4_expected_sarsa.ipynb](4_expected_sarsa.ipynb)          |
| [Dyna-Q](5_dyna_q.ipynb)                                        | Model-Based   | Combines direct RL (Q-learning) with planning via a learned environment model, improving sample efficiency.                                                       | [5_dyna_q.ipynb](5_dyna_q.ipynb)                           |
| [REINFORCE](6_reinforce.ipynb)                                | Policy-Based  | A Monte Carlo policy gradient method that directly optimizes a parameterized policy based on complete episode returns.                                                  | [6_reinforce.ipynb](6_reinforce.ipynb)                   |
| [Proximal Policy Optimization (PPO)](7_ppo.ipynb)               | Actor-Critic  | State-of-the-art, stabilizes policy updates via clipped surrogate objective. Balances exploration and exploitation more efficiently.                         | [7_ppo.ipynb](7_ppo.ipynb)                               |
| [Advantage Actor-Critic (A2C)](8_a2c.ipynb)                      | Actor-Critic  | Uses a critic to estimate advantages, reducing variance compared to REINFORCE. Synchronous updates.                                                                 | [8_a2c.ipynb](8_a2c.ipynb)                               |
| [Asynchronous Advantage Actor-Critic (A3C)](9_a3c.ipynb)        | Actor-Critic  | An asynchronous version of A2C, using multiple workers to collect data and update the global network. (See `a3c_training.py`)                                | [9_a3c.ipynb](9_a3c.ipynb)                               |
| [Deep Deterministic Policy Gradient (DDPG)](10_ddpg.ipynb)         | Actor-Critic  | Uses a separate action function to estimate Q-values, allowing operation in continuous action spaces.                                               | [10_ddpg.ipynb](10_ddpg.ipynb)         |
| [Soft Actor-Critic (SAC)](11_sac.ipynb)        | Actor-Critic  | Is a state-of-the-art **off-policy actor-critic** algorithm designed for **continuous action spaces**, building significantly upon ideas from DDPG, TD3, and incorporating the principle of **maximum entropy reinforcement learning**. | [11_sac.ipynb](11_sac.ipynb)         |
| [Trust Region Policy Optimization (TRPO)](12_trpo.ipynb)        | On Policy  | Imposes a limit on how much the policy distribution can change in a single step.   | [12_trpo.ipynb](12_trpo.ipynb)         |
| [Deep Q-Network (DQN)](13_dqn.ipynb)        | Value Based   | Combines Q-learning with deep neural networks to handle complex problems with high-dimensional state spaces.   | [13_dqn.ipynb](13_dqn.ipynb)         |
| [Multi-Agent Deep Deterministic Policy Gradient (MADDPG)](14_maddpg.ipynb)        | Actor-Critic  | Extends the DDPG algorithm designed to address the non-stationarity problem in multi-agent settings.   | [14_maddpg.ipynb](14_maddpg.ipynb)         |
| [QMIX: Monotonic Value Function Factorization for MARL](15_qmix.ipynb)        | On-Policy Actor-Critic  |  is a value-based MARL algorithm designed for cooperative tasks with value function factorization.  | [15_qmix.ipynb](15_qmix.ipynb)         |
| [Hierarchical Actor-Critic (HAC)](16_hac.ipynb)        | Hierarchical  |  It decomposes long, complex tasks into manageable sub-problems, making learning feasible where standard RL might fail.  | [16_hac.ipynb](16_hac.ipynb)         |
| [Monte Carlo Tree Search (MCTS)](17_mcts.ipynb)        | Planning  |    Is a best-first search algorithm guided by the results of random simulations (Monte Carlo rollouts).  | [17_mcts.ipynb](17_mcts.ipynb)         |
| [PlaNet (Deep Planning Network)](18_planet.ipynb)        | Planning  |   Is a model-based RL agent that learns a world model from experience and uses this model to plan future actions.  | [18_planet.ipynb](18_planet.ipynb)         |

Each algorithm has its own Jupyter Notebook (`.ipynb`) file with a detailed explanation and implementation.

## 📚 RL Cheat Sheet

Complementing the detailed notebooks, a comprehensive **[RL Cheat Sheet](RL_Cheat_Sheet.md)** is included in this repository. It serves as a quick reference guide covering:

*   Core RL Concepts (MDPs, Bellman Equations, etc.)
*   Algorithm Summaries (Core Idea, Math, Pseudocode)
*   Key Hyperparameters and Tuning Tips
*   Pros & Cons and Use Cases
*   Code Snippets for key update rules

➡️ **[View the RL Cheat Sheet here](RL_Cheat_Sheet.md)**

## 🛠️ Installation and Setup

Follow these steps to get started:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/fareedkhan-dev/all-rl-algorithms.git
    cd all-rl-algorithms
    ```

2.  **Create a virtual environment (Recommended):**

    ```bash
    python -m venv .venv-all-rl-algos
    source .venv-all-rl-algos/bin/activate  # Linux/macOS
    .venv-all-rl-algos\Scripts\activate  # Windows
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```
4.  **Multiprocessing in A3C:** Please run `a3c_training.py` in the terminal instead of the jupyter notebook to avoid any complication from multiprocessing.

## 🧑‍🏫 How to Use This Repo: A Learning Guide

1.  **Start with the Basics (`1_simple_rl.ipynb`):** This notebook introduces fundamental RL concepts like states, actions, rewards, and policies.
2.  **Explore Core Algorithms:** Dive into the individual notebooks for Q-Learning (`2_q_learning.ipynb`), SARSA (`3_sarsa.ipynb`), and REINFORCE (`6_reinforce.ipynb`). Understand their update rules, strengths, and limitations.
3.  **Analyze the Code:** Carefully read the code comments, which explain the purpose of each function and variable.
4.  **Experiment!:** This is where the real learning begins. Try these:
    *   Change hyperparameters (learning rate, discount factor, exploration rate) and observe the effect on learning curves.
    *   Modify the environment (e.g., change the grid size, add obstacles) and see how the algorithms adapt.
    *   Extend the algorithms (e.g., implement epsilon decay, add a baseline to REINFORCE).
5.  **Consult the Cheat Sheet:** Refer to the **[RL Cheat Sheet](RL_Cheat_Sheet.md)** for quick summaries, formulas, and pseudocode while studying the notebooks.
6.  **Tackle Advanced Methods:** Gradually work through the more complex notebooks on DQN (`13_dqn.ipynb`), Actor-Critic (`8_a2c.ipynb`), PPO (`7_ppo.ipynb`), Model-Based RL with PlaNet (`18_planet.ipynb`), and multi-agent learning with MADDPG (`14_maddpg.ipynb`) and QMIX (`15_qmix.ipynb`).
7.  **Run the A3C Implementation:** Due to complexities with multiprocessing in Jupyter Notebooks, the A3C implementation is in `a3c_training.py`. Run it from the command line: `python a3c_training.py`

## 🖼️ What You'll See: Visualizing Learning

Each notebook includes visualizations to help you understand the agent's behavior:

-   **Learning Curves:** Plots of episode rewards, episode lengths, and loss functions.
-   **Q-Table Visualizations:** Heatmaps to visualize Q-values across the state space (tabular methods).
-   **Policy Grids:** Arrows showing the learned policy (action choice) in each state.
-   **More Advanced Visualizations:** Visualizations may depend on each particular algorithm.

## ⚠️ Disclaimer: Bugs and Incomplete Implementations

This repository is primarily for learning! While effort has been taken, some notebooks (especially the more complex ones like HAC) may contain bugs, incomplete implementations, or simplifications for clarity. If you find any issues, feel free to create a pull request.

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

*   **Fix Bugs:** Found an error or a way to improve the code? Submit a pull request!
*   **Improve Explanations:** Clarify confusing sections in the notebooks or add more helpful comments.
*   **Add More Algorithms:** Implement algorithms currently marked as "Planned."
*   **Create More Visualizations:** Develop insightful visualizations to better understand the learning process.
*   **Add More Environment Examples:** Implement known RL tasks.
*   Please follow the project's coding style and documentation guidelines.
*   Create a new issue to discuss your contribution before starting work.