import parl
import paddle
import numpy as np


class Car(parl.Agent):


    def __init__(self, algorithm, act_dim, e_greed=0.1, e_greed_decrement=0):
        super(Car, self).__init__(algorithm)
        assert isinstance(act_dim, int)
        self.act_dim = act_dim

        self.global_step = 0
        self.update_target_steps = 200

        self.e_greed = e_greed
        self.e_greed_decrement = e_greed_decrement

    def sample(self, obs):
        #Sample一个action用于探索

        sample = np.random.random()
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)
        else:
            if np.random.random() < 0.01:
                act = np.random.randint(self.act_dim)
            else:
                act = self.predict(obs)
        self.e_greed = max(0.01, self.e_greed - self.e_greed_decrement)   #e-greedy，同Q-learning
        return act

    def predict(self, obs):
        #predict一个action

        obs = paddle.to_tensor(obs, dtype='float32')
        pred_q = self.alg.predict(obs)      #选最大的Q值
        act = int(pred_q.argmax())
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        #更新model

        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, axis=-1)
        reward = np.expand_dims(reward, axis=-1)
        terminal = np.expand_dims(terminal, axis=-1)

        obs = paddle.to_tensor(obs, dtype='float32')
        act = paddle.to_tensor(act, dtype='int32')
        reward = paddle.to_tensor(reward, dtype='float32')
        next_obs = paddle.to_tensor(next_obs, dtype='float32')
        terminal = paddle.to_tensor(terminal, dtype='float32')
        loss = self.alg.learn(obs, act, reward, next_obs, terminal)
        return float(loss)