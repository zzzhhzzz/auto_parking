import gym
import numpy as np
import argparse
import parking_env
from parl.utils import logger, ReplayMemory
from car_model import CarModel
from car_agent import Car
from parl.env import CompatWrapper
from parl.algorithms import DQN
#from DQN import DQN

LEARN_FREQ = 5  # training frequency
MEMORY_SIZE = 200000
MEMORY_WARMUP_SIZE = 200
BATCH_SIZE = 64
LEARNING_RATE = 0.0005
GAMMA = 0.99


# train an episode
def run_train_episode(agent, env, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)
        next_obs, reward, done, _ = env.step(action)
        rpm.append(obs, action, reward, next_obs, done)
        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            # s,a,r,s',done
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample_batch(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs, batch_done)

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# evaluate 5 episodes
def run_evaluate_episodes(agent, eval_episodes=5, render=False):
    env = gym.make("parking_env-v0", render=False, mode="2")
    env = CompatWrapper(env)

    eval_reward = []
    for i in range(eval_episodes):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)


def main():
    env = gym.make("parking_env-v0", mode="2",render=False)
    env = CompatWrapper(env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n
    logger.info('obs_dim {}, act_dim {}'.format(obs_dim, act_dim))

    # set action_shape = 0 while in discrete control environment
    rpm = ReplayMemory(MEMORY_SIZE, obs_dim, 0)

    # build an agent
    model = CarModel(obs_dim=obs_dim, act_dim=act_dim)
    alg = DQN(model, gamma=GAMMA, lr=LEARNING_RATE)
    agent = Car(
        alg, act_dim=act_dim, e_greed=0.1, e_greed_decrement=1e-6)

    # warmup memory
    while len(rpm) < MEMORY_WARMUP_SIZE:
        run_train_episode(agent, env, rpm)

    max_episode = args.max_episode

    # start training
    episode = 0
    while episode < max_episode:
        # train
        for i in range(10):
            total_reward = run_train_episode(agent, env, rpm)
            episode += 1
            logger.info('episode:{}    e_greed:{}  '.format(
                episode, agent.e_greed))
        # test
        eval_reward = run_evaluate_episodes(agent, render=False)
        logger.info('episode:{}    e_greed:{}   Test reward:{}'.format(
            episode, agent.e_greed, eval_reward))

    #保存模型参数
    save_path = './DQN_model/model.ckpt'
    agent.save(save_path)

    save_inference_path = './inference_model'
    input_shapes = [[None, env.observation_space.shape[0]]]
    input_dtypes = ['float32']
    agent.save_inference_model(save_inference_path, input_shapes, input_dtypes)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--max_episode',
        type=int,
        default=10,
        help='stop condition: number of max episode')
    args = parser.parse_args()

    main()
