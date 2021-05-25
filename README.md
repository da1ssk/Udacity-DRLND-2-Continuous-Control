# Udacity-DRLND-2-Continuous-Control

In this project, we train a double-jointed arm to reach target locations.

The simulation environment is based on Unity Environment. The code is available [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/python/unityagents)

<img src="Reacher_with_learned_weights.gif" width=480>

### Rewards
- +0.1 for touching the target
- 0 otherwise

### States
There are 33 states in this environment. Those consist of position, rotation, velocity, and angular velocities of the arm.

### Actions
Each action is a vector with 4 numbers, corresponding to torque applicable to two joints. Every entry in the action vector is clipped between -1 and 1.

### Criteria
I chose Option 1: Solve the First Version.
If the average score of the latest 100 episodes exceeds +30, it is considered solved.

## Getting started
Load `Continuous_Control.ipynb` in Jupyter notebook, then follow the instruction.

## Report
The report can be found [here](Report.md)
