import parl
import paddle.fluid as fluid
import copy
from parl import layers

import parl
class DQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ DQN algorithm
        Args:
            model: Q函数的前向网络结构
            act_dim: action空间的维度，即有几个action
            gamma: reward的衰减因子
            lr: learning rate 学习率.
        """
        self.model = model 
        self.target_model = copy.deepcopy(model)    # 创建target Q模型，直接从model复制给target

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim = act_dim      
        self.gamma = gamma
        self.lr = lr

    def predict(self, obs):     # 使用 current Q network 获取所有action的 Q values

        return self.model.value(obs) 

    def learn(self, obs, action, reward, next_obs, terminal):
         
        # 使用DQN算法更新self.model的value网络
        
        # 计算target_Q
        next_pred_value = self.target_model.value(next_obs)     # 获取 target Q network 的所有action的 Q values
        best_v = layers.reduce_max(next_pred_value, dim=1)      # 获取最大的Q值
        best_v.stop_gradient = True         # 阻止梯度传递
        terminal = layers.cast(terminal, dtype='float32') 
        target = reward + (1.0 - terminal) * self.gamma * best_v #判断是否该终止

        pred_value = self.model.value(obs)  

        # 计算Q(s,a)
        action_onehot = layers.one_hot(action, self.act_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)

        # 计算 Q(s,a) 与 target_Q的MSE均方差，得到loss
        cost = layers.square_error_cost(pred_action_value, target)
        cost = layers.reduce_mean(cost) 
        optimizer = fluid.optimizer.Adam(learning_rate=self.lr)       # 使用Adam优化器
        optimizer.minimize(cost)
        return cost

    def sync_target(self):
        #把 self.model 的模型参数值同步到 self.target_model
        
        self.model.sync_weights_to(self.target_model) # 直接调用API 更新 Target Q
