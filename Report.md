# Continuous Control project report

## Learning Algorithm

I used the code from [`ddpg-bipedal`](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal) in the Udacity Deep Reinforcement Learning Nanodegree repo as the base.

The training algorithm thus is the Deep Deterministic Policy Gradient (DDPG).

### Agent ([`ddpg_agent.py`](ddpg_agent.py))
I use the `Agent` class from the `ddpg-bipedal` repo with minor changes.

The only change I made was the `step` function to take `LEARN_EVERY` into account. Instead of learning every step, it waits until the modulo of step count divided by `LEARN_EVERY` equals zero.

Here are the hyperparameters.
- `BUFFER_SIZE` = int(1e6)  # replay buffer size
- `BATCH_SIZE` = 64         # minibatch size
- `GAMMA` = 0.99            # discount factor
- `TAU` = 1e-3              # for soft update of target parameters
- `LR_ACTOR` = 1e-4         # learning rate of the actor 
- `LR_CRITIC` = 1e-4        # learning rate of the critic
- `WEIGHT_DECAY` = 0.0      # L2 weight decay
- `LEARN_EVERY` = 10        # frequency to learn

### Model architecture ([`model.py`](model.py))
I use the `Actor` and the `Critic` class from the `ddpg-bipedal` repo with minor changes.

The only change I made is to add one hidden layer to the Actor network before the last fc unit.

The Actor is a neural network with fully connected layers. The first and second hidden layer is followed by a rectifier nonlinearity (that is, `max(0,x)`).

- Input: 33 nodes corresponding to each state element
- First layer: 256 nodes
- Second layer: 256 nodes
- Third layer: 128 nodes
- Output: 4 nodes corresponding to the value of each action

The Critic is also a neural network with fully connected layers as well.
- Input: 33 nodes corresponding to each state element
- First layer: 256 nodes
- Second layer: 256 + 4 nodes where the 4 nodes are for the sampled action
- Third layer: 128 nodes
- Output: single float Q value for the state + action pair

## Plot of Rewards
Here are the latest average scores for every 100 episodes. 
```
Episode 100	Average Score: 0.78
Episode 200	Average Score: 2.37
Episode 300	Average Score: 4.12
Episode 400	Average Score: 6.01
Episode 500	Average Score: 8.29
Episode 600	Average Score: 7.93
Episode 700	Average Score: 7.69
Episode 800	Average Score: 8.30
Episode 900	Average Score: 10.17
Episode 1000	Average Score: 10.37
Episode 1100	Average Score: 10.63
Episode 1200	Average Score: 11.11
Episode 1300	Average Score: 12.40
Episode 1400	Average Score: 13.80
Episode 1500	Average Score: 14.13
Episode 1600	Average Score: 14.91
Episode 1700	Average Score: 15.71
Episode 1800	Average Score: 17.08
Episode 1900	Average Score: 17.91
Episode 2000	Average Score: 20.56
Episode 2100	Average Score: 20.52
Episode 2200	Average Score: 22.84
Episode 2300	Average Score: 24.01
Episode 2400	Average Score: 25.72
Episode 2500	Average Score: 26.31
Episode 2600	Average Score: 27.22
Episode 2700	Average Score: 29.79
Episode 2800	Average Score: 29.18
Episode 2900	Average Score: 30.93
Episode 3000	Average Score: 30.66
```

Here is the plot of rewards over episodes.

![rewards](https://user-images.githubusercontent.com/1985201/119428521-07d21400-bcdb-11eb-8f9c-c762c6e62bbe.png)

It took 2900 episodes to pass the criteria.

## Using the learned weight for visualization
I visualized the agent by loading the learned actor's weight (`checkpoint_actor.pth`).

Here's the result. The score for this episode was 30.15.

<img src="Reacher_with_learned_weights.gif" width=480>

I ran the following code for this visualization, in place of the section `1.0.3  3. Take Random Actions in the Environment` in `Continuous_Control.ipynb`.

```
import torch
from ddpg_agent import Agent
 
agent = Agent(state_size=state_size, action_size=action_size, random_seed=2)

# load the trained actor network
agent.actor_local.load_state_dict(torch.load("checkpoint_actor.pth", map_location={'cuda:0': 'cpu'}))

env_info = env.reset(train_mode=False)[brain_name]
states = env_info.vector_observations
scores = np.zeros(num_agents)
while True:
    actions = agent.act(states)
    env_info = env.step(actions)[brain_name]
    next_states = env_info.vector_observations
    rewards = env_info.rewards
    dones = env_info.local_done
    scores += env_info.rewards
    states = next_states
    if np.any(dones):
        break
print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
```

## Ideas for Future Work

### Tuning Hyperparameters and models
In the benchmark implementation section in the course, I noticed that it reaches the solution much faster (less than 200 episodes for the single agent). That’s the evidence that there are better hyperparameters and network models. I would love to know Udacity's solution.

### Work on the 20 agents
It would be challenging and fun to work on the second option. However, I feel like I'd like to know what would be the best solution for the single-agent case before going further. 

### Development cycle
Again, the workspace Udacity provides was a huge pain. Namely, it takes too much time to: 
- load the workspace
- install (!?) python
- load the unity environment
- train, even with GPU support

Another big problem is that when I stop the training, I need to refresh the workspace every time, again repeating the above time-consuming process.
There is no point using the workspace unless the cloud environment is better than a local one. This is future work for Udacity.

## Development Environment
First I used the Udacity workspace in the course for this project. As I almost used up the GPU usage, I decided to build my local environment on my Ubuntu with Nvidia GeForce GTX 1080 GPU. It turned out it trains much faster than the Udacity environment, which leads me to a solution with much faster iterations.

### Use of Reacher_One_Linux_NoVis
I used the single reacher version with no visualization for the sake of learning speed. I couldn't find the link to the headless binary in the course material, but I found it from a random former student's report, for example, [here](https://github.com/bobiblazeski/reacher).
Please add the link to the course material for future students.