#%matplotlib inline - used for colab.research.google.com
import numpy as np
import matplotlib.pyplot as plt

#number of arms for each bandit
ARMS = 10
#number of runs
EXP = 2000
#number of steps
N = 1000
class Bandit:
    '''
    @arm_values: the values for each action using the normal distribution
    @K: action count
    @estimates: estimation for each action
    @best_action: the best action at the current time
    '''
    def __init__(self):
        self.arm_values = np.random.randn(ARMS)
        self.K = np.zeros(ARMS)
        self.estimates = np.zeros(ARMS)
        self.best_action = np.argmax(self.arm_values)
    '''
    Get an action (exploration vs exploitation)

    #eps: the probability for exploration
    '''
    def epsGreedy(self,eps):
        rand_num = np.random.rand()
        if eps>rand_num:
            return np.random.choice(ARMS)
        else:
            q_best = np.max(self.estimates)
            return np.random.choice(np.where(self.estimates == q_best)[0])
    '''
    Calculate reward for an action using some random noise

    @action: current action
    '''
    def reward(self,action):
        return self.arm_values[action] + np.random.randn()
    '''
    Update estimates based on reward using step_size function

    @action: currenta action
    @reward: reward correspoding to the best action
    '''
    def updateEstimates(self,action,reward):
        self.K[action] += 1
        #sample average function
        alpha = 1.0/self.K[action] 
        self.estimates[action] += alpha * (reward - self.estimates[action])

'''
Calculate the average reward and percentage of optimal actions 

@bandit: object Bandit to represent each bandit
@epsilon: the probability to explore the action space
@best_action_counts: the array that represents the optimal actions
@i: index that represents corresponding experiment (run)
'''
def calculateExp(bandit,epsilon,best_action_counts,i):
    reward_array = []
    best_action_ =  []
    for j in range(N):
        #get some action and some reward for this corresponding action
        action = bandit.epsGreedy(epsilon)
        reward = bandit.reward(action)
        bandit.updateEstimates(action,reward)
        #if the optimal action is selected then update the best_action_counts array to 1
        if action == bandit.best_action:
          best_action_counts[j, i] = 1
        reward_array.append(reward)
    mean_best_action_counts = best_action_counts.mean(axis=1)
    return (np.array(reward_array), mean_best_action_counts)

'''
Main Function 
'''
if __name__ == '__main__':
    #initializing the best action counts to 2d array 
    best_action_counts0 = np.zeros((N, EXP))
    best_action_counts01 = np.zeros((N, EXP))
    best_action_counts001 = np.zeros((N, EXP))
    #initializing the average reward arays to 0  
    average_eps0 = np.zeros(N)
    average_eps01 = np.zeros(N)
    average_eps001 = np.zeros(N)

    for i in range(EXP):
        # average for greedy method (eps = 0.0)
        bandit = Bandit()
        avg0,act0 = calculateExp(bandit,0.0,best_action_counts0,i)
        average_eps0 += avg0

        #average for eps-greedy method (eps = 0.1)
        bandit = Bandit()
        avg01,act01 = calculateExp(bandit,0.1,best_action_counts01,i)
        average_eps01 += avg01

        #average for eps-greedy method (eps = 0.01)
        bandit = Bandit()
        avg001,act001 = calculateExp(bandit,0.01,best_action_counts001,i)
        average_eps001 += avg001

    #calculating the average reward
    average_eps0 /= np.float(EXP)
    average_eps01 /= np.float(EXP)
    average_eps001 /= np.float(EXP)

    #Plotting the graph for average reward
    plt.plot(average_eps0,label="eps = 0.0")
    plt.plot(average_eps001,label="eps = 0.01")
    plt.plot(average_eps01,label="eps = 0.1")
    plt.ylim(0,1.5)
    plt.xlabel("steps")
    plt.ylabel("average reward")
    plt.legend()
    plt.show()

    #Plotting the graph for optimal action selection
    plt.plot(act0,label="eps = 0.0")
    plt.plot(act001,label="eps = 0.01")
    plt.plot(act01,label="eps = 0.1")
    plt.ylim(0,1)
    plt.xlabel("steps")
    plt.ylabel("% optimal action")
    plt.legend()
    plt.show()
