import gym
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from collections import deque

class DQNAgent:

	def __init__(self,env):
		self.env = env
		
		#Hyperparameters
		self.memory = deque(maxlen=2000)
		self.discount = 0.85
		self.epsilon = 1.0 #it is not a hyperparameter
		self.epsilon_min = 0.01
		self.epsilon_decay = 0.995
		self.learning_rate = 0.005
		self.tau = .125

		#Creating neural network for action-value approximation
		self.model = self.create_model()
		#Two models are created to stabilize learning
		self.target_model = self.create_model()

	def create_model(self):
		#Creating network with 3 hidden layers
		model = Sequential()
		state_shape = self.env.observation_space.shape
		model.add(Dense(24, input_dim=state_shape[0], activation="relu"))
		model.add(Dense(48, activation="relu"))
		model.add(Dense(24, activation="relu"))
		model.add(Dense(self.env.action_space.n))
		model.compile(loss="mean_squared_error", optimizer=Adam(lr=self.learning_rate))
		return model

	#Defining our policy / e-greedy
	def act(self,state):
		#Chance of acting greedily will increase every time we act
		self.epsilon *= self.epsilon_decay
		self.epsilon = max(self.epsilon_min, self.epsilon)
		#Acting reandom
		if np.random.random() < self.epsilon:
			return self.env.action_space.sample()
		#Acting greedily
		return np.argmax(self.model.predict(state)[0]) #Returns the index of maximum action-value

	#Recording the steps that we acted with e-greedy policy
	#(s,a,r,s',isFinished)
	def remember(self, state, action, reward, new_state, done):
		self.memory.append([state, action, reward, new_state, done])

	#Replay experience by sampling it
	def replay(self):
		batch_size = 32
		#if there is not enough experience, do nothing
		if len(self.memory) < batch_size:
			return

		#Gets (s,a,r,s',done) pairs randomly
		samples = random.sample(self.memory, batch_size)

		for sample in samples:
			state, action, reward, new_state, done = sample
			target = self.target_model.predict(state)
			#for final act, action-value is just reward. there is no extra future reward that we need to predict
			if done:
				target[0][action] = reward

			else:

				q_future = max(self.target_model.predict(new_state)[0])
				target[0][action] = reward + q_future*self.discount
			#not training our target model. we avoid chasing non-stationary target
			self.model.fit(state, target, epochs=1, verbose=0)


	#We will synchronize target model and actual model in every k episode
	def target_train(self):
		weights = self.model.get_weights()
		target_weights = self.target_model.get_weights()
		for i in range(len(target_weights)):
			target_weights[i] = weights[i]*self.tau + target_weights[i]*(1-self.tau)

		self.target_model.set_weights(target_weights)

	#Saving network weights to disk
	def save_model(self, fn):
		self.model.save(fn)


def main():
	print("It started")
	env = gym.make("MountainCar-v0")
	gamma = 0.9
	epsilon = 0.95
	trials = 1000
	trial_len = 500

	dqn_agent = DQNAgent(env=env)
	steps = []
	for trial in range(trials):
		cur_state = env.reset().reshape(1,2)

		for step in range(trial_len):
			action = dqn_agent.act(cur_state)
			new_state, reward, done, info = env.step(action)
			new_state = new_state.reshape(1,2)
			dqn_agent.remember(cur_state, action, reward, new_state,done)

			dqn_agent.replay()
			dqn_agent.target_train()

			cur_state = new_state
			if done:
				break

		if step>=199:
			print("Episode: " + str(trial) + " failed.")
		else:
			print("Epside: " + str(trial) + " completed.")

if __name__ == "__main__":
	main()