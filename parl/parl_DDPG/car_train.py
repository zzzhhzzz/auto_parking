import gym
import argparse
import numpy as np
from parl.utils import logger, summary, ReplayMemory
from parl.env import ActionMappingWrapper, CompatWrapper
from car_agent import Car
from car_model import Carmodel2
from parl.algorithms import DDPG
#gym 要求 0.26.0 高于DQN要求0.21.0
WARMUP_STEPS = 1e4
EVAL_EPISODES = 5
MEMORY_SIZE = int(1e6)
BATCH_SIZE = 100
GAMMA = 0.99
TAU = 0.005
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3
EXPL_NOISE = 0.1  


#训练
def run_train_episode(agent, env, rpm):
    action_dim = env.action_space.shape[0]
    obs = env.reset()
    done = False
    episode_reward, episode_steps = 0, 0

    while not done:
        episode_steps += 1
        #random action
        if rpm.size() < WARMUP_STEPS:
            action = np.random.uniform(-1, 1, size=action_dim)
        else:
            action = agent.sample(obs)

        #action
        next_obs, reward, done, _ = env.step(action)
        terminal = float(done) if episode_steps < env._max_episode_steps else 0

        #经验回放
        rpm.append(obs, action, reward, next_obs, terminal)
        obs = next_obs
        episode_reward += reward

        if rpm.size() >= WARMUP_STEPS:
            batch_obs, batch_action, batch_reward, batch_next_obs, batch_terminal = rpm.sample_batch(
                BATCH_SIZE)
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_terminal)

    return episode_reward, episode_steps



def run_evaluate_episodes(agent, env, eval_episodes):
    avg_reward = 0.
    for _ in range(eval_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    return avg_reward


def main():
    logger.info("------------------ DDPG ---------------------")
    logger.info('Env: {}, Seed: {}'.format(args.env, args.seed))
    logger.info("---------------------------------------------")
    logger.set_dir('./{}_{}'.format(args.env, args.seed))

    env = gym.make(args.env)
    env = CompatWrapper(env)
    env = ActionMappingWrapper(env)

    env.seed(args.seed)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    #模型初始化
    model = Carmodel2(obs_dim, action_dim)
    algorithm = DDPG(
        model, gamma=GAMMA, tau=TAU, actor_lr=ACTOR_LR, critic_lr=CRITIC_LR)
    agent = Car(algorithm, action_dim, expl_noise=EXPL_NOISE)
    rpm = ReplayMemory(
        max_size=MEMORY_SIZE, obs_dim=obs_dim, act_dim=action_dim)

    total_steps = 0
    test_flag = 0
    while total_steps < args.train_total_steps:
        #训练
        episode_reward, episode_steps = run_train_episode(agent, env, rpm)
        total_steps += episode_steps

        summary.add_scalar('train/episode_reward', episode_reward, total_steps)
        logger.info('Total Steps: {} Reward: {}'.format(
            total_steps, episode_reward))

        #评估
        if (total_steps + 1) // args.test_every_steps >= test_flag:
            while (total_steps + 1) // args.test_every_steps >= test_flag:
                test_flag += 1
            avg_reward = run_evaluate_episodes(agent, env, EVAL_EPISODES)
            summary.add_scalar('eval/episode_reward', avg_reward, total_steps)
            logger.info('Evaluation over: {} episodes, Reward: {}'.format(
                EVAL_EPISODES, avg_reward))

    save_inference_path = './inference_model'
    input_shapes = [[None, env.observation_space.shape[0]]]
    input_dtypes = ['float32']
    agent.save_inference_model(save_inference_path, input_shapes, input_dtypes,
                               model.actor_model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--env", default="parking_env-v0")

    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument(
        "--train_total_steps",
        default=5e6,
        type=int,
        help='Max time steps to run environment')
    parser.add_argument(
        '--test_every_steps',
        type=int,
        default=int(5e3),
        help='The step interval between two consecutive evaluations')
    args = parser.parse_args()

    main()