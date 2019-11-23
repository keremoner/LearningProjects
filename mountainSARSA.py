import gym
import numpy as np
import time
from matplotlib import pyplot as plt

env = gym.make("MountainCar-v0")
# Oyunu resetler ve aynı zamanda ilk state observation'larını verir
# env.reset()

# Hyperparameters
learning_rate = 0.01
discount = 0.95
episodes = 25000
SHOW_EVERY = 1000
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = episodes // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

# Q-table boyutumuzu belirliyoruz, yani toplam 400 farklı state var
osSize = [20, 20]
# Asıl observation değerlerimizden 0,20 arasına maplemek için gereken step değeri hesabı
osStepSize = (env.observation_space.high - env.observation_space.low) / osSize

# 20x20x3'lük bir array oluşturduk. 3 action olduğu için x3 var
q_table = np.random.uniform(low=-2, high=0, size=(osSize + [env.action_space.n]))


def stateMap(state):
    mappedState = (state - env.observation_space.low) / osStepSize
    return tuple(mappedState.astype(np.int))  # Bulduğumuz arrayi tamsayı arrayine dönüştürdü

tstart = time.time()
timeAxis = []
stepAxis = []
avgAxis = []
totalStep = 0
for episode in range(episodes):

    mappedState = stateMap(env.reset())
    done = False
    action = 0
    if np.random.random() > epsilon:
        action = np.argmax(q_table[mappedState])
    else:
        action = np.random.randint(0, env.action_space.n)
    step = 1
    while not done:
        # Greedy action, it gives the index of the greatest action-value pair
        new_state, reward, done, _ = env.step(action)
        new_mappedState = stateMap(new_state)
        totalStep+=1
        # Render the game
        """
        if episode % SHOW_EVERY == 0:
            env.render()
        """


        # Check if game is finished after we made our action
        if not done:
            # Prediction of next action value if we acted greedily
            if np.random.random() > epsilon:
                newaction = np.argmax(q_table[new_mappedState])
            else:
                newaction = np.random.randint(0, env.action_space.n)
            future_q = q_table[new_mappedState + (action,)]

            # Current q value with respect to our action
            current_q = q_table[mappedState + (action,)]
            # q-learning algorithm to update our q value
            new_q = current_q + learning_rate*(reward + discount*future_q - current_q)
            q_table[mappedState + (action,)] = new_q

        elif new_state[0] >= env.goal_position:
            q_table[mappedState + (action,)] = 0
        action = newaction
        mappedState = new_mappedState
        step+=1

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value
    """
    if step>=199:
        print("Episode: " + str(episode) + "failed")
    else:
        print("Episode: " + str(episode) + "SUCCESS step: " + str(step))
    """
    tend = time.time()
    timeAxis.append((tend - tstart)/60)
    avgAxis.append(totalStep/(episode+1))
    stepAxis.append(step)
    print(episode)

plt.plot(timeAxis,avgAxis)
plt.plot(timeAxis,stepAxis)
plt.savefig("mountainSARSA.png")
env.close()
