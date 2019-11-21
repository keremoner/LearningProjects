import gym
import numpy as np

env = gym.make("MountainCar-v0")
#Oyunu resetler ve aynı zamanda ilk state observation'larını verir
#env.reset()

#Hyperparameters
learning_rate = 0.01
discount = 0.95
episodes = 25000
SHOW_EVERY = 1000
epsilon = 1
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = episodes//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING - START_EPSILON_DECAYING)


#Q-table boyutumuzu belirliyoruz, yani toplam 400 farklı state var
osSize = [20,20]
#Asıl observation değerlerimizden 0,20 arasına maplemek için gereken step değeri hesabı
osStepSize = (env.observation_space.high - env.observation_space.low)/osSize

#20x20x3'lük bir array oluşturduk. 3 action olduğu için x3 var
q_table = np.random.uniform(low=-2, high=0, size=(osSize+[env.action_space.n]))

def stateMap(state):
	mappedState = (state-env.observation_space.low)/osStepSize 
	return tuple(mappedState.astype(np.int)) #Bulduğumuz arrayi tamsayı arrayine dönüştürdü
for episode in range(episodes):

	mappedState = stateMap(env.reset())
	done = False

	while not done:
		#Greedy action, it gives the index of the greatest action-value pair
		if np.random.random() > epsilon:
			action = np.argmax(q_table[mappedState])
		else:
			action = np.random.randint(0, env.action_space.n)

		new_state, reward, done, _ = env.step(action)
		new_mappedState = stateMap(new_state)


		#Render the game
		if episode % SHOW_EVERY == 0:
			env.render()
			print(episode)

		#Check if game is finished after we made our action
		if not done:
			#Prediction of next action value if we acted greedily
			max_future_q = np.max(q_table[new_mappedState])

			#Current q value with respect to our action
			current_q = q_table[mappedState + (action,)]
			#q-learning algorithm to update our q value
			new_q = (1- learning_rate) * current_q + learning_rate*(reward + discount*max_future_q)
			q_table[mappedState + (action,)] = new_q

		elif new_state[0] >= env.goal_position:
			q_table[mappedState + (action,)] = 0

		mappedState = new_mappedState

	if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
		epsilon -= epsilon_decay_value



env.close()

