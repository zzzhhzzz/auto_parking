import parl
import paddle
import numpy as np
#gym 要求 0.26.0 高于DQN要求0.21.0


class Car(parl.Agent):
    def __init__(self, algorithm, act_dim, expl_noise=0.1):
        assert isinstance(act_dim, int)
        super(Car, self).__init__(algorithm)

        self.act_dim = act_dim
        self.expl_noise = expl_noise

        self.alg.sync_target(decay=0)

    def sample(self, obs):
        action_numpy = self.predict(obs)
        action_noise = np.random.normal(0, self.expl_noise, size=self.act_dim)
        action = (action_numpy + action_noise).clip(-1, 1)
        return action

    def predict(self, obs):
        obs = paddle.to_tensor(obs.reshape(1, -1), dtype='float32')
        action = self.alg.predict(obs)
        action_numpy = action.cpu().numpy()[0]
        return action_numpy

    def learn(self, obs, action, reward, next_obs, terminal):
        terminal = np.expand_dims(terminal, -1)
        reward = np.expand_dims(reward, -1)

        obs = paddle.to_tensor(obs, dtype='float32')
        action = paddle.to_tensor(action, dtype='float32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        critic_loss, actor_loss = self.alg.learn(obs, action, reward, next_obs,
                                                 terminal)
        return critic_loss, actor_loss