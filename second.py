import numpy as np
import matplotlib
#%matplotlib inline - used only in colab.research.google.com
import matplotlib.pyplot as plt

# world height
WORLD_HEIGHT = 4
# world width
WORLD_WIDTH = 12
# all possible actions
VALID_ACTIONS = [0, 1, 2, 3]
# initial state action pair values
START_STATE = [3, 0]
GOAL_STATE = [3, 11]
AVERAGE_RUNS = 10
#number of episodes 
EPISODES = 500
#number of runs (experiments)
RUNS = 10
#penalty for cliff walking
CLIFF_PENALTY = 100

'''
Calculates the next state and reward corresponding to it

@state: the current state
@action: the selected action
@return next state and reward corresponding to this action-state pair
'''
def calculateState(state, action):
    i, j = state
    if action == VALID_ACTIONS[0]:
        next_state = [max(i - 1, 0), j]
    elif action == VALID_ACTIONS[2]:
        next_state = [i, max(j - 1, 0)]
    elif action == VALID_ACTIONS[3]:
        next_state = [i, min(j + 1, WORLD_WIDTH - 1)]
    elif action == VALID_ACTIONS[1]:
        next_state = [min(i + 1, WORLD_HEIGHT - 1), j]
    else:
        assert False
    reward = -1
    if (action == VALID_ACTIONS[1] and i == 2 and 1 <= j <= 10) or (
        action == VALID_ACTIONS[3] and state == START_STATE):
        reward = - CLIFF_PENALTY
        next_state = START_STATE

    return next_state, reward
'''
Choose an action based on epsilon greedy algorithm

@state: Current state
@q_value : value for the state action pair
'''
def epsGreedy(state, q_value,eps = 0.1):
    rand_num = np.random.rand()
    if eps > rand_num:
        return np.random.choice(VALID_ACTIONS)
    else:
        state_values = q_value[state[0], state[1], :]
        return np.random.choice([state_action for state_action, state_value in enumerate(state_values) if state_value == np.max(state_values)])

'''
@q_value: values for state action pair
@algo: Type of the algorithm
@learn_rate: Learning rate for the algorithm
'''
def computeOptimal(q_value, algo = 'SARSA', learn_rate = 0.1, gamma = 1):
  #SARSA algorithm
  if(algo == 'SARSA'):
    state = START_STATE
    current_action = epsGreedy(state, q_value)
    rewards = 0.0
    while state != GOAL_STATE:
        next_state, reward = calculateState(state, current_action)
        next_action = epsGreedy(next_state, q_value)
        rewards += reward
        target = q_value[next_state[0], next_state[1], next_action]
        target *= gamma
        # SARSA update rule
        q_value[state[0], state[1], current_action] += learn_rate * (reward + target - q_value[state[0], state[1], current_action])
        state = next_state
        current_action = next_action
    return rewards
  #Q_Learning algorithm
  elif(algo == 'Q_LEARNING'):
    state = START_STATE
    rewards = 0.0
    while state != GOAL_STATE:
        current_action = epsGreedy(state, q_value)
        next_state, reward = calculateState(state, current_action)
        rewards += reward
        # Q-Learning update rule
        q_value[state[0], state[1], current_action] += learn_rate * (reward + gamma * np.max(q_value[next_state[0], next_state[1], :]) - q_value[state[0], state[1], current_action])
        state = next_state
    return rewards

#Main Function
if __name__ == '__main__':
    rewardsSarsa = np.zeros(EPISODES)
    rewardsQlearning = np.zeros(EPISODES)
    for r in range(RUNS):
        q_sarsa = np.zeros((WORLD_HEIGHT, WORLD_WIDTH, 4))
        q_q_learning = np.copy(q_sarsa)
        for i in range(0, EPISODES):
            rewardsSarsa[i] += max(computeOptimal(q_sarsa, 'SARSA'), - CLIFF_PENALTY)
            rewardsQlearning[i] += max(computeOptimal(q_q_learning, 'Q_LEARNING'), - CLIFF_PENALTY)

    # averaging over independent runs
    rewardsSarsa = rewardsSarsa/ RUNS
    rewardsQlearning = rewardsQlearning/ RUNS

    # averaging over successive episodes
    smoothedRewardsSarsa = np.copy(rewardsSarsa)
    smoothedRewardsQLearning = np.copy(rewardsQlearning)
    for i in range(AVERAGE_RUNS, EPISODES):
        smoothedRewardsSarsa[i] = np.mean(rewardsSarsa[i - AVERAGE_RUNS: i + 1])
        smoothedRewardsQLearning[i] = np.mean(rewardsQlearning[i - AVERAGE_RUNS: i + 1])

    # draw reward curves
    plt.plot(smoothedRewardsSarsa, label='Sarsa')
    plt.plot(smoothedRewardsQLearning, label='Q-Learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards during episode')
    plt.legend()
    plt.show()
