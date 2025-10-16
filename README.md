# Q-Learning on Taxi-v3  
**Assignment 2 — Reinforcement Learning (CSCN 8020)**  

This project implements a Q-Learning agent for the Taxi-v3 environment from Gymnasium.  
The goal is to train the agent to pick up passengers, drop them off at their destinations, and maximize cumulative rewards through experience.

---

## Overview
The notebook explores how different hyperparameters (learning rate, exploration rate, and discount factor) affect the learning process.  
The agent starts with no knowledge and gradually learns an optimal policy through trial and error using the ε-greedy strategy.

We test several parameter combinations, compare their results, and finally simulate the best-performing model in a rendered environment to visualize its behavior.

---

## Objectives
- Implement Q-Learning from scratch using a tabular approach.  
- Test and compare multiple hyperparameter combinations:
  - Learning Rate (α)
  - Exploration Rate (ε)
  - Discount Factor (γ)
- Analyze the performance of each run based on total rewards and convergence.  
- Save and visualize results.  
- Simulate the trained agent in Taxi-v3 using the best Q-table.

---
.
├── assignment2_notebook.ipynb
├── assignment2_utils.py
├── results/
│   ├── q_table_best.npy
│   ├── metrics_baseline.csv
│   ├── plot_baseline.png
│   └── assignment2_report.pdf  
|── README.md
└──requirements.txt

---

Author

Eris Leksi
CSCN 8020 — Artificial Intelligence for Business Decisions and Transformation
Conestoga College, 2025

---
References

Gymnasium Documentation

Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.)

CSCN 8020 Course Materials and Assignment 2 Guidelines

---
## Setup and Requirements
Before running the notebook, install the required packages:

```bash
pip install gymnasium numpy pandas matplotlib

