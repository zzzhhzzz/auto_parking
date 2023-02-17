import argparse
import os

import gym
import parking_env
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


env = gym.make("parking_env-v0", render=False, mode="2")
#env = gym.make("parking_env-v0", render=True, mode="3")
env.reset()

model = DQN('MlpPolicy', env, verbose=1, seed=0)
logger = configure("./model", ["stdout", "csv", "tensorboard"])
model.set_logger(logger)
checkpoint_callback = CheckpointCallback(save_freq=8016, save_path="./model", name_prefix='stable_baselines3_dqn')
model.learn(total_timesteps=8016, callback=checkpoint_callback)
model.save("./model")
del model

"""
env = gym.make("parking_env-v0", render=False, mode="2")
#env = gym.make("parking_env-v0", render=True, mode="3")
obs = env.reset()
#model = DQN.load("./model", env=env)

episode_return = 0
for i in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    episode_return += reward
    if done:
        for j in range(10000000):
            reward += 0.0001
        break

env.close()

print(f'episode return: {episode_return}')
"""
