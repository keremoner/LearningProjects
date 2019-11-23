import gym
import numpy as np
import time
from matplotlib import pyplot as plt

class TDAgent:
    def __init__(self, env):
        self.env = env
        self.state_shape = self.env.observation_space.shape
        # Hyperparameters
        self.discount = 0.99
        self.epsilon = 1.0  # it is not a hyperparameter
        self.epsilon_min = 12499
        self.epsilon_decay = 1/(12499.0)
        self.alpha = 0.01
        self.lambdav = 0.9
        self.osSize = [20, 20]
        # Asıl observation değerlerimizden 0,20 arasına maplemek için gereken step değeri hesabı
        self.osStepSize = (env.observation_space.high - env.observation_space.low) / self.osSize

        # 20x20x3'lük bir array oluşturduk. 3 action olduğu için x3 var
        self.action_table = np.random.uniform(low=-2, high=0, size=(self.osSize + [self.env.action_space.n]))
        self.eTraces = np.zeros(self.osSize+[self.env.action_space.n])
    def stateMap(self, state):
        mappedState = (state - self.env.observation_space.low) / self.osStepSize
        return tuple(mappedState.astype(np.int))  # Bulduğumuz arrayi tamsayı arrayine dönüştürdü

    def act(self, state):
        # Chance of acting greedily will increase every time we act
        #self.epsilon = max(self.epsilon_min, self.epsilon)
        # Acting reandom
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, self.env.action_space.n)
        else:
            action = np.argmax(self.action_table[state])
        # Acting greedily
        # Returns the index of maximum action-value
        return action

    def train(self, state, action, reward, newstate, done):
        value = self.action_table[state + (action,)]
        newact = self.act(newstate)
        newValue = self.action_table[newstate + (newact,)]
        if not done:
            target = reward + self.discount*newValue
        else:
            target = reward
        error = target - value
        self.eTraces[state + (action, )] += 1
        self.action_table[state + (action, )] += error*self.alpha*self.eTraces[state + (action, )]
        self.eTraces = self.discount*self.lambdav*self.eTraces
        return newact


def main():
    print("It started")
    env = gym.make("MountainCar-v0")
    trials = 25000
    trial_len = 500
    tdagent = TDAgent(env=env)
    bestMan = 200
    timeAxis = []
    avgAxis = []
    stepAxis = []
    tstart = time.time()
    totalStep = 0
    for trial in range(trials):
        state = tdagent.stateMap(env.reset())
        action = tdagent.act(state)
        tdagent.eTraces = np.zeros(tdagent.osSize+[tdagent.env.action_space.n])
        for step in range(trial_len):
            totalStep+=1
            new_state, reward, done, info = env.step(action)
            new_state = tdagent.stateMap(new_state)
            action = tdagent.train(state,action,reward,new_state,done)
            state = new_state
            """
            if trial%1000==0:
                env.render()
            """
            if done:
                break

        """if step >=199:
            print("Episode: " + str(trial) + " failed" + str(tdagent.epsilon))
        else:
            print("Episode" + str(trial) + " SUCCESSSS!")
        """
        if 12500>=trial>=1:
            tdagent.epsilon -= tdagent.epsilon_decay
        tend = time.time()
        timeAxis.append((tend-tstart)/60)
        stepAxis.append(step)
        avgAxis.append(totalStep/(trial+1))


    plt.plot(timeAxis, stepAxis)
    plt.plot(timeAxis, avgAxis)
    plt.savefig('mountainTDLambda.png')


if __name__ == "__main__":
    main()
