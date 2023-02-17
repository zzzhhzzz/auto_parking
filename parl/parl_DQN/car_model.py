import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import parl


class CarModel(parl.Model):
    """
    初始化Model
    定义三层全连接网络用于DQN
    输入数据维度为obs_dim；
    隐藏神经元数量为128；
    输出维度为act_dim，即行动空间
    """
    def __init__(self, obs_dim, act_dim):
        super(CarModel, self).__init__()
        hid1_size = 128
        hid2_size = 128
        self.fc1 = nn.Linear(obs_dim, hid1_size)
        self.fc2 = nn.Linear(hid1_size, hid2_size)
        self.fc3 = nn.Linear(hid2_size, act_dim)

    def forward(self, obs):
        h1 = F.relu(self.fc1(obs))
        h2 = F.relu(self.fc2(h1))
        Q = self.fc3(h2)
        return Q

    """
    定义前向网络x为网络的输入
    使用relu作为两层网络激活函数
    """