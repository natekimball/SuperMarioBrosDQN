import numpy as np
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D
from keras.optimizers import Adam
from collections import deque
import random

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, Flatten, Dense, AveragePooling2D, GlobalAveragePooling2D, MaxPooling2D

def residual_block(inputs, filters):
    x = Conv2D(filters, kernel_size=(3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('elu')(x)
    x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([inputs, x])
    x = Activation('elu')(x)
    return x

def build_model(input_shape, output_size, name='medium'):
    if name == 'big':
        inputs = Input(shape=input_shape)
        x = Conv2D(128, kernel_size=(4, 4), strides=(1, 1))(inputs)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)
        x = residual_block(x, 128)
        x = AveragePooling2D(pool_size=(2, 2))(x)
        x = residual_block(x, 128)
        x = AveragePooling2D(pool_size=(2,2))(x)
        x = Flatten()(x)
        x = Dense(256)(x)
        x = Dense(output_size)(x)
        model = Model(inputs=inputs, outputs=x)
    elif name == 'medium':
        model = Sequential([
            Conv2D(64, kernel_size=(4, 4), strides=(1, 1), input_shape=input_shape),
            BatchNormalization(),
            Activation('elu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), strides=(1, 1), input_shape=input_shape),
            BatchNormalization(),
            Activation('elu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(256, kernel_size=(3, 3), strides=(1, 1), input_shape=input_shape),
            BatchNormalization(),
            Activation('elu'),
            GlobalAveragePooling2D(),
            Dense(output_size)
        ])
    elif name == 'small':
        model = Sequential([
            Conv2D(64, kernel_size=(4, 4), strides=(1, 1), input_shape=input_shape),
            BatchNormalization(),
            Activation('relu'),
            Conv2D(128, kernel_size=(3, 3), strides=(1, 1), input_shape=input_shape),
            BatchNormalization(),
            Activation('relu'),
            Flatten(),
            Dense(output_size),
        ])
    elif name == 'extra-small':
        model = Sequential([
            Conv2D(64, kernel_size=(4, 4), strides=(1, 1), input_shape=input_shape),
            BatchNormalization(),
            Activation('relu'),
            Flatten(),
            Dense(output_size),
        ])
    else:
        raise ValueError('Invalid model')
    return model

class DQNAgent:
    def __init__(self, state_shape, action_size, max_mem=4000, epsilon_decay=0.9999, gamma=0.99, epsilon_greedy=True, model_path=None, initial_epsilon=1.0, learning_rate=0.001, save_dir='models/mario', model='medium'):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=max_mem)
        self.gamma = gamma  # discount rate
        self.epsilon = initial_epsilon if not epsilon_greedy else 0 # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = epsilon_decay
        self.optimizer = Adam(learning_rate=learning_rate)
        if model_path is not None:
            self.model = self._load_model(model_path)
        else:
            self.model = self._build_model(model)
        self.epsilon_greedy = epsilon_greedy
        self.save_dir = save_dir

    def _build_model(self, model_name):
        model = build_model(self.state_shape, self.action_size, model_name)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model
    
    def _load_model(self, filename):
        return keras.models.load_model(filename)
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if self.epsilon_greedy and np.random.rand() <= self.epsilon:
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
        
        if self.epsilon_greedy and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self):
        self.model.save(self.save_dir)