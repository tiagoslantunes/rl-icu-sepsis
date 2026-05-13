# Sepsis Treatment Optimization via Reinforcement Learning

## 📌 Project Overview
Sepsis is a life-threatening condition in Intensive Care Units (ICUs) that requires dynamic, high-stakes decision-making.  
This project applies Reinforcement Learning (RL) to learn optimal treatment policies directly from patient data: the objective is to balance the administration of vasopressors and intravenous fluids—maximizing patient survival rates while minimizing unnecessary treatment intensity.

## Project Structure
The project is divided into two main configurations of increasing complexity:

- Configuration A (Tabular RL): a discrete Markov Decision Process (MDP) with 716 states and 25 actions; we implement classical algorithms (Policy Iteration, Q-Learning, SARSA) to establish a strong baseline and understand the core environment dynamics.
- Configuration B (Continuous Deep RL): a clinically grounded environment mapping to a 47-dimensional continuous feature space — this phase tests the agent's robustness against real-world clinical failure modes:
  - Episodic observation noise (monitor malfunctions).
  - Episodic missing observations (unavailable lab results).
  - Acute clinical events (sudden, unpredictable patient deterioration).

## How We Are Approaching This
To ensure high-quality delivery, the project follows an iterative, three-phase methodology:
1. Baselines & Tabular: validate the environment setup, implement model-based and model-free tabular agents and establish baseline survival metrics.
2. Deep RL & Stress Testing: transition to continuous-state agents to handle high-dimensional data, rigorously testing policy robustness against the injected clinical failure wrappers.
3. Clinical Translation & Reporting: translate technical metrics into clinical insights — we will finalize a creative extension (e.g., Explainable AI or reward shaping) and focus the final report on actionable, comparative analysis.
