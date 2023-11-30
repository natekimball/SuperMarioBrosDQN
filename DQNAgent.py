import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_shape, action_size, max_mem=4000, epsilon_decay=0.9999, gamma=0.99, test_mode=False, model_path=None):
        self.state_shape = state_shape
        print(state_shape)
        self.action_size = action_size
        self.memory = deque(maxlen=max_mem)
        self.gamma = gamma  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.learning_rate = 0.001
        if model_path is not None:
            self.model = self._load_model(model_path)
        else:
            self.model = self._build_model()
        self.test_mode = test_mode

    def _build_model(self):
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='selu', padding='same', input_shape=self.state_shape),
            # Conv2D(64, kernel_size=(3, 3), activation='selu', padding='same'),
            Flatten(),
            # Dense(64, activation='selu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def _load_model(self, filename):
        return keras.models.load_model(filename)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if not self.test_mode and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.reshape(state, [1, 240, 256, 3]), verbose=0)[0]
        return np.argmax(q_values)

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states = []
        future_states = []
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            future_states.append(next_state)
        future_values = np.max(self.model.predict(np.array(future_states), verbose=0), axis=1)
        # states = np.array([s[0] for s in minibatch])
        # future_states = np.array([f[3] for f in minibatch])
        # future_values = np.max(self.model.predict(future_states, verbose=0), axis=1)
        states = np.array(states)
        targets = np.array(self.model.predict(states, verbose=0))
        for i, (state, action, reward, next_state, done) in enumerate(minibatch):
            if done:
                targets[i][action] = reward
            else:
                targets[i][action] = reward + future_values[i] * self.gamma
        
        self.model.fit(states, targets, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, filename):
        self.model.save(filename)

class MockDQNAgent:
    def __init__(self, state_shape, action_size, max_mem=10):
        self.action_size = action_size

    def _build_model(self):
        pass

    def remember(self, state, action, reward, next_state, done):
        pass

    def act(self, state):
        return random.randrange(self.action_size)

    def replay(self, batch_size):
        pass

# def train_dqn(episode):
#     loss = []
#     agent = DQNAgent(state_size, action_size)
#     for e in range(episode):
#         state = get_state()  # Define your own function to get the state
#         state = np.reshape(state, [1, state_size])
#         done = False
#         i = 0
#         while not done:
#             action = agent.act(state)
#             next_state, reward, done = step(action)  # Define your own function to take a step
#             next_state = np.reshape(next_state, [1, state_size])
#             agent.remember(state, action, reward, next_state, done)
#             state = next_state
#             if done:
#                 print("episode: {}/{}, score: {}, e: {:.2}".format(e, episode, i, agent.epsilon))
#                 break
#             if len(agent.memory) > batch_size:
#                 agent.replay(batch_size)
#         if e % 10 == 0:
#             agent.save("./save/mario-dqn.h5")