import parl
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
#gym 要求 0.26.0 高于DQN要求0.21.0
'''
Model of DDPG:  Actor and Critic

'''


class Carmodel2(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Carmodel2, self).__init__()
        self.actor_model = Actor(obs_dim, action_dim)
        self.critic_model = Critic(obs_dim, action_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, action):
        return self.critic_model(obs, action)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class Actor(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(obs_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, action_dim)

    def forward(self, obs):
        a = F.relu(self.l1(obs))
        a = F.relu(self.l2(a))
        return paddle.tanh(self.l3(a))


class Critic(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(obs_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, 1)

    def forward(self, obs, action):
        q = F.relu(self.l1(obs))
        q = F.relu(self.l2(paddle.concat([q, action], 1)))
        return self.l3(q)