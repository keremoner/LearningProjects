from tictactoe import game
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque
import time
from matplotlib import pyplot as plt


class DQNAgent:

    def __init__(self, env):
        self.env = env
        # Hyperparameters
        self.memory = deque(maxlen=10000)
        self.discount = 0.99
        self.epsilon = 1.0  # it is not a hyperparameter
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999539589
        self.learning_rate = 0.002
        self.tau = .125

        # Creating neural network for action-value approximation
        self.model = self.create_model()

        # Trained model
        # self.model.load_weights('trainNetworkInEPS218.h5')

        # Two models are created to stabilize learning
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

    def create_model(self):
        # Creating network with 2 hidden layers
        model = Sequential()
        model.add(Dense(24, input_shape=[9], activation="relu"))
        model.add(Dense(48, activation="relu"))
        model.add(Dense(9, activation="linear"))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    # Defining our policy / e-greedy
    def act(self, state):
        # Chance of acting greedily will increase every time we act
        action = 0
        #self.epsilon = max(self.epsilon_min, self.epsilon)
        # Acting reandom
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, 9)
        else:
            action = np.argmax(self.model.predict(state)[0])
        # Acting greedily
        # Returns the index of maximum action-value
        return action

    # Recording the steps that we acted with e-greedy policy
    # (s,a,r,s',isFinished)
    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    # Replay experience by sampling it
    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size:
            return

        samples = random.sample(self.memory, batch_size)

        states = []
        new_states = []
        for sample in samples:
            state, action, reward, new_state, done = sample
            states.append(state)
            new_states.append(new_state)

        newArray = np.array(states)
        states = newArray.reshape(batch_size, 9)

        newArray2 = np.array(new_states)
        new_states = newArray2.reshape(batch_size, 9)

        targets = self.model.predict(states)
        new_state_targets = self.target_model.predict(new_states)

        i = 0
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = targets[i]
            if done:
                target[action] = reward
            else:
                Q_future = max(new_state_targets[i])
                target[action] = reward + Q_future * self.discount
            i += 1

        self.model.fit(states, targets, epochs=1, verbose=0)

    # We will synchronize target model and actual model in every k episode
    def target_train(self):

        self.target_model.set_weights(self.model.get_weights())

    # Saving network weights to disk
    def save_model(self, fn):
        self.model.save(fn)


def main():
    print("It started")
    env = game()
    trials = 10000
    trial_len = 20
    dqn_agent = DQNAgent(env=env)

    bestMan = 200
    timeAxis = []
    stepAxis = []
    avgAxis = []
    totalStep = 0
    tstart = time.time()
    for trial in range(trials):

        cur_state = env.reset().reshape(1, env.obsSpace)
        step = 0
        while True:
            step+=1
            action = dqn_agent.act(cur_state)
            new_state, reward, done = env.step(action)
            new_state = new_state.reshape(1, env.obsSpace)
            totalStep+=1

            if trial % 500 == 0:
                env.render()

            if (not done[1]) and done[0]:
                reward += 10
            if done[0]:
                dqn_agent.remember(cur_state, action, reward, new_state, done)
                dqn_agent.replay()
                break

            else:
                possible = []
                for i in range(9):
                    if env.table[i] ==0:
                        possible.append(i)
                new_state, rreward, done = env.step(possible[np.random.randint(0,len(possible))])

            if done[0]:
                break
            else:
                new_state = new_state.reshape(1, env.obsSpace)
                dqn_agent.remember(cur_state, action, reward, new_state, done)
                dqn_agent.replay()

            cur_state = new_state

        """
        if step >= 199:
            print("Episode: " + str(trial) + " failed.")
        else:
            print("Epside: " + str(trial) + " completed." + str(step))
            if step < bestMan:
                print("New best score!!")
                bestMan = step
                dqn_agent.model.save('./MountainCarModels/mountainDDQN.h5')
        print(str(trial))
        tend = time.time()
        timeAxis.append((tend-tstart)/60)
        avgAxis.append(totalStep/(trial+1))
        stepAxis.append(step)
        """

        if trial==9999:
            dqn_agent.model.save('./ttt10k.h5')
        dqn_agent.target_train()
        dqn_agent.epsilon *= dqn_agent.epsilon_decay
        print(str(trial))
    """
    plt.plot(timeAxis,stepAxis)
    plt.plot(timeAxis,avgAxis)
    plt.savefig("mountainDDQNsmallnetwork.png")
    """
    dqn_agent.model.save('./ttt50k.h5')
if __name__ == "__main__":
    main()
    """
    env = game()
    trials = 1000
    trial_len = 20
    dqn_agent = DQNAgent(env=env)
    dqn_agent.model.load_weights('./ttt25k.h5')
    dqn_agent.epsilon = 0
    state = env.reset().reshape(1,env.obsSpace)
    env.render()
    while True:
        action = dqn_agent.act(state)
        newstate, reward, done = env.step(action)
        env.render()
        action = int(input())
        newstate, reward, done = env.step(action)
        env.render()

        if done[0]:
            state = env.reset().reshape(1,env.obsSpace)
        else:
            state = newstate.reshape(1,env.obsSpace)
"""