from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
import gym
from DQNAgent import DQNAgent
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('SuperMarioBros-v3', apply_api_compatibility=True)#, render_mode="human")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

state_shape = env.observation_space.shape
action_size = env.action_space.n
agent = DQNAgent(state_shape, action_size)
TIMESTEPS = 1000000
batch_size = 32

high_scores = []
record = 0
done = True
for step in range(TIMESTEPS):
    if step % 100 == 0:
        high_scores.append(record)
    if done:
        state, info = env.reset()
    action = agent.act(state)
    next_state, reward, terminated, truncated, info = env.step(action)
    record = max(record, info['score'])
    done = terminated or truncated
    agent.remember(state, action, reward, next_state, done)
    state = next_state
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    
env.close()
agent.save('mario-v1')

plt.figure(figsize=(10, 5))
plt.plot(high_scores, label='Highest Score at time step')
plt.xlabel('Every 100th time step')
plt.ylabel('Score')
plt.legend()
plt.savefig('mario_scores.png')
plt.show()

# # EPISODES = 1
# # scores = []
# # moving_avg_scores = []
# # for e in range(EPISODES):
# #     state, info = env.reset()
# #     done = False
# #     score = 0
# #     for time in range(5000):
# #         action = agent.act(state)
# #         next_state, reward, terminated, truncated, info = env.step(action)
# #         done = terminated or truncated
# #         agent.remember(state, action, reward, next_state, done)
# #         state = next_state
# #         score = info['score']
# #         if done:
# #             break
# #         if len(agent.memory) > batch_size:
# #             agent.replay(batch_size)
# #     print(f"episode: {e}/{EPISODES}, score: {score}, ε: {agent.epsilon:.2}")
# #     scores.append(score)
# #     moving_avg_scores.append(np.mean(scores[-100:]))

# # scores = []
# # moving_avg_scores = []
# # e = 1
# # done = True
# # for time in range(1000000):
# #     if done:
# #         state, info = env.reset()
# #     action = agent.act(state)
# #     next_state, reward, terminated, truncated, info = env.step(action)
# #     done = terminated or truncated
# #     agent.remember(state, action, reward, next_state, done)
# #     state = next_state
# #     if len(agent.memory) > batch_size:
# #         agent.replay(batch_size)
# #     if done:
# #         score = info['score']
# #         print(f"episode: {e}, score: {score}, ε: {agent.epsilon:.2}, time-step: {time}")
# #         scores.append(score)
# #         moving_avg_scores.append(np.mean(scores[-100:]))
# #         break
# #     e += 1
