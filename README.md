# 自动泊车强化学习训练及小车实地部署教程
![](https://ai-studio-static-online.cdn.bcebos.com/f16f60e3d8bf495699d5f07e0e3df850d7a69e94e4e442b09200e8e64a3939f2)

实现自动泊车的思路如上所示，接下来将分别介绍如何实地部署，模拟强化学习，并将两者结合

实地部署（实时确定位姿）部分[b站视频教程](https://www.bilibili.com/video/BV1ge411N7Wx?p=3&vd_source=df155d7dda1976141acbe4e9a5979b74)和[github项目地址](https://github.com/zzzhhzzz/pose_estimate)

基于飞桨强化学习训练自动泊车部分[b站视频教程](https://www.bilibili.com/video/BV1V8411E7bJ/?spm_id_from=333.999.0.0&vd_source=df155d7dda1976141acbe4e9a5979b74)和[github项目地址](https://github.com/zzzhhzzz/auto_parking)

使用AIstudio试运行项目：https://aistudio.baidu.com/aistudio/projectdetail/5442330

##  一、实地部署
图片
### 1.小车运动控制
![](https://ai-studio-static-online.cdn.bcebos.com/74fad631f94e4469b31f9517a446b946ad1a35c80bff413997d3acceeb726657)

设置频率，信号通道，分辨率等，将引脚和通道进行绑定，根据小车电机连接管脚，设置输出管脚与输入管脚，通过写入占空比的方式控制小车前进或后退。

小车的左转右转需要在小车前进的过程中使用舵机调整角度，舵机同样通过写入pwm信号对角度进行控制，最终将小车前进、后退、左转、右转四个动作分别封装成函数。电脑连接小车wifi，通过向小车IP地址发送request请求实现小车运动控制。

*具体开发板代码可在[arduino小车自动驾驶项目](https://github.com/zzzhhzzz/automatic_drive_arduino)中找到。*

### 2.实时确定位姿
![](https://ai-studio-static-online.cdn.bcebos.com/8bb02606bacb44e8b23bcef266a142718582facad006432aafd3e5b4fa260b16)

#### 2.1.手机相机标定去畸变
##### 2.1.1.标定原理：

”张正友标定法”是指张正友教授在[Zhang Z. A flexible new technique for camera calibration[J]. IEEE Transactions on pattern analysis and machine intelligence, 2000, 22(11): 1330-1334.](https://ieeexplore.ieee.org/abstract/document/888718/)文章中提出的单平面棋盘格的摄像机标定方法，此方法被广泛应用于计算机视觉方面，以下对其标定原理进行简单介绍。

![](https://ai-studio-static-online.cdn.bcebos.com/96c9769e9f434814959a2caea1f21f22ef93a35a14a947d693af548c4a22b290)

设三维世界坐标的点为$X=[X,Y,Z,1]^{T}$,二维相机平面像素坐标为$m=[u,v,1]^{T}$,标定用的棋盘格平面到图像平面的单应性关系为：$s_{0}m=K[R,T]X$,其中s为尺度因子，K为摄像机内参数$K={\left[\begin{array}{l l l}{\alpha}&{\gamma}&{u_{0}}\\ {0}&{\beta}&{v_{0}}\\ {0}&{0}&{1}\end{array}\right]}$，R为旋转矩阵，T为平移向量,把$K[r1, r2, t]$叫做单应性矩阵H，即
$s\left[\begin{array}{c}{{u}}\\ {{v}}\\ {{1}}\end{array}\right]=H\left[\begin{array}{c}{{X}}\\ {{Y}}\\ {{1}}\end{array}\right]$
每个单应性矩阵能提供两个方程，而内参数矩阵包含5个参数，要求解，至少需要3个单应性矩阵。为了得到三个不同的单应性矩阵，我们使用至少三幅棋盘格平面的图片进行标定。$B=K^{-T}K^{-1}=\left[\begin{array}{l l l}{{B_{11}}}&{{B_{12}}}&{{B_{13}}}\\ {{B_{21}}}&{{B_{22}}}&{{B_{23}}}\\ {{B_{31}}}&{{B_{32}}}&{{B_{33}}}\end{array}\right]$计算得到B，然后通过cholesky分解，得到相机的内参数矩阵K，外参由${r l}{\lambda={\frac{1}{s}}={\frac{1}{\|A^{-1}h_{1}\|}}={\frac{1}{\|A^{-1}h_{2}\|}}}$式计算得到，其中${r_{1}={\frac{1}{\lambda}}K^{-1}h_{1}}$,${r_{2}={\frac{1}{\lambda}}K^{-1}h_{2}}$,${t=\lambda K^{-1}h_{3}}$。但上述的推导结果是基于理想情况下的解，由于可能存在高斯噪声，所以通过多采集照片，使用最大似然估计进行优化。在去畸变上，张氏标定法只关注了影响最大的径向畸变，可由$k=[k_{1}\:k_{2}]^{T}=(D^{T}D)^{-1}D^{T}d$计算得到畸变系数k。

##### 2.1.2.标定步骤：
1. 打印标定板（此处演示标定板大小为8x8）
2. 使用待标定相机对标定板拍摄若干张图片（10-20张）
3. 在图片中检测特征点（Harris特征）
4. 利用解析解估算方法计算出相机内外参数
5. 根据极大似然估计策略，设计优化目标并实现参数的refinement。

##### 2.1.3.示例代码：

```python
import cv2
import numpy as np
import glob
from matplotlib import pyplot as plt
# 设置寻找亚像素角点的参数，采用的停止准则是最大循环次数30和最大误差容限0.001
criteria = (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001)
 
def getPoints(data_path, m, n):
    # 获取标定板角点的位置
    objp = np.zeros((m * n, 3), np.float32)
    objp[:, :2] = np.mgrid[0:m, 0:n].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
 
    obj_points = []  # 存储3D点
    img_points = []  # 存储2D点
 
    images = glob.glob(data_path)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        ret, corners = cv2.findChessboardCorners(gray, (m, n), None)
        # print(ret)
        if ret:
            obj_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)  # 在原角点的基础上寻找亚像素角点
            #print(corners2)
            if [corners2]:
                img_points.append(corners2)
            else:
                img_points.append(corners)
            #cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.drawChessboardCorners(img, (m, n), corners, ret)
            #cv2.imshow('img', img)
            plt.imshow(img)
            #cv2.waitKey(0)
 
    #print(len(img_points))
    cv2.destroyAllWindows()
    return obj_points, img_points, size
 
obj_points, img_points, size = getPoints("标定测试图片.jpg", 7, 7)
# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, size, None, None)
print("ret:", ret)
print("mtx:\n", mtx)  # 内参数矩阵
print("dist:\n", dist)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs:\n", rvecs)  # 旋转向量  # 外参数
print("tvecs:\n", tvecs)  # 平移向量  # 外参数
```

#### 2.2.利用opencv ArUco标记确定位姿

##### 2.2.1 方法简介

ArUco方法由S. Garrido-Jurado等人于2014年在[Automatic generation and detection of highly reliable fiducial markers under occlusion](https://www.researchgate.net/publication/260251570_Automatic_generation_and_detection_of_highly_reliable_fiducial_markers_under_occlusion)一文中提出，在[opencv的官方文档](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)中可以找到ArUco的使用方法。
 ArUco 标记是由宽黑色边框和确定其标识符（id）的内部二进制矩阵组成的正方形标记，被广泛用来增加从二维世界映射到三维世界时的信息量，便于发现二维世界与三维世界之间的投影关系，其可以根据对象大小和场景选择自身大小，并且可以迅速检测相应位姿。

*（aruco包含在opencv-contrib-python包中，aistudio中默认配置opencv-python且无法卸载重装opencv-contrib-python，所以此部分代码无法在notebook中运行，需要在本地pip安装opencv-contrib-python==3.4.8.29后运行。代码在aruco文件夹中）*
![](https://ai-studio-static-online.cdn.bcebos.com/7f1af2a841f04c8daf766cf560622701dbef9a374ad846b4a9cd9ff48c327303)

##### 2.2.2 实现实时定位

为了实现实时定位，可以使用ip摄像头软件，电脑通过ip地址可将手机摄像图像实时读取处理。在地图一角和小车顶部分别贴上aruco标记，即可通过两者相对位置和x轴y轴的偏移，迅速确定小车在地图上的相对位置。

#### 2.3.结合yolov5完善位置确定

靠此方法虽能准确确定小车姿态，但只能获取到小车大体位置，为了完善小车边界位置的检测，可以使用yolov5目标检测。通过自己标注数据集完成训练，大约100张左右即可达到不错的效果。通过结合两个方法，即可实时定位小车的确切位姿。*（yolov5基于PyTorch框架，无法在notebook中运行，此部分代码在[github中开源](https://github.com/zzzhhzzz/pose_estimate/tree/master/yolov5_detect%20direction)）*

yolov5训练结果：

![](https://ai-studio-static-online.cdn.bcebos.com/c62b5eb1b5a44a399ad37b76e4536ebe67fc147f447f4df3b7c9af197451bf40)

![](https://ai-studio-static-online.cdn.bcebos.com/a8210171738b42a2bbd9b3af968236b359e1021081a841e48028663f0dc20cfc)

##  二、强化学习训练

### 1.强化学习环境搭建

根据之前的工作，结合opencv的aruco标记与yolov5目标检测，确定小车的位姿和小车与车库的相对位置，输出用来搭建强化学习环境的位置参数。利用[pybullet](https://pybullet.org/wordpress/)仿真模拟小车运动，使用pybullet包中自带的小车模型，计算出它与实际小车的大小比例大致为1：2，在[gym](https://github.com/openai/gym)中完成对实际场景成比例缩放的强化学习环境搭建。

![](https://ai-studio-static-online.cdn.bcebos.com/3803ca8397804d51bc30bb1ea94df62182e8a83d82764082b6237dc1950341f2)

```shell
cd ~
cd parl
pip install -e parking_env
cd ~
pip install parl==2.1.1
pip install cloudpickle==2.2.0
pip install pybullet==3.2.5
pip install gym==0.21.0
pip install importlib-metadata==4.0
```

### 2.基于paddle的强化学习

在搭建好强化学习环境后，即可使用飞桨框架完成强化学习任务。针对此任务可以使用飞桨搭建DQN和DDPG两个模型用于强化学习。下面对两个模型的实现进行讲解。

飞桨提供了[parl](https://github.com/PaddlePaddle/PARL)库用于完成强化学习任务。其核心思想是把强化学习算法框架拆分成为三个嵌套的model、algorithm、agent。

![](https://ai-studio-static-online.cdn.bcebos.com/a3ae55fcac354d66afa2740d87e12cddcb8b8f82343d4bbe971e80ba2c8342f5)


model部分用来定义有神经网络部分的网络结构。可以是dqn中的q值网络或是policy中的策略网络。在修改网络结构时可以直接在model中修改。其输入是当前环境状态State。algorithm即是具体的算法，用损失函数来更新model。agent负责和环境做交互，在交互过程中把生成的数据提供给Algorithm来更新模型 (Model)。

#### 2.1 使用parl实现DQN训练自动泊车

![](https://ai-studio-static-online.cdn.bcebos.com/c63ca80842314a748d28b891e7217ba685eaa477f10f4eeea8f51482d08d7c32)

首先介绍DQN算法实现。其核心思想是将神经网络融入Q-learning。它的创新点在于引入了经验回放，打乱样本相关性，提高样本利用率。经验回放可使用以下代码实现

```python
import random
import collections
import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)

#也可直接调用Parl库中的API：from parl.utils import ReplayMemory
```

parl也提供了此算法的API。DQN除了引入经验回放，它还创建了定期更新的targetQ，用于固定Q目标，提高算法平稳性。使用parl的实现流程如下。

![](https://ai-studio-static-online.cdn.bcebos.com/f50dd46b5a4e459bb94cc24eba2fc4343a17f8c87cd1477b8c53bde824495438)

首先是model部分，它实现了一个model class，model class会继承parl.model，只需要实现一个value，forward函数，即输出价值，输出q值。这里定义三层全连接网络，配置relu激活函数。最后输出的神经元的个数为动作数。到时候只要调用model里面的value函数，就可以输入observation，然后输出和行动空间对应的q值

```python
# DQN Model部分代码
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
```

第二部分是**algorithm**。在输入里面把刚刚有定义好的model输入进来，直接deep copy就可以拿到一个网络结构跟model一模一样的**target model**。

![](https://ai-studio-static-online.cdn.bcebos.com/facd3ef198d04564b7a607ba8a0dd1edab570519ca7c405bb077e801fe285625)

然后使用sync_target函数去实现模型参数的同步，即selfmodel同步到target model。只需要调用parl实现好的modal层面参数拷贝的API，即可做定期的一个参数的同步。Dqn要实现的predic函数也很简单，它直接返回model输出的值就可以，即调用value函数输入observation，输出action个数的list。其中learn函数总体的逻辑分为三部分：
1. 第一部分我们计算Q目标targetq。target q的计算需要用到q-learning公式$y_{j}={\left\{\begin{array}{l l}{r_{j}}&{}\\ {r_{j}+\gamma\operatorname*{max}_{a^{\prime}}Q(\phi_{j+1},a^{\prime};\theta)}\end{array}\right.}$，当没有下一步状态时，目标值就是当前单步的reward。在这一部分我们需要**阻止梯度传递**，因为第三部分中使用的paddle优化器默认会将所有loss function的相关参数一起优化，但DQN中target_model的参数是固定不动的，所以这里必须阻止梯度传递。

2. 第二部分计算q（s，a）即输出的预测值，计算时将action先转成onehot向量，再和输出的q值按位相乘再相加，就可以得到 q（s，a）
3. 第三部分得到targetq和q（s，a）之后，就可以调用Parl的均方差的函数计算出loss。然后放到adam优化器里面，自动执行minimize操作。

![](https://ai-studio-static-online.cdn.bcebos.com/357dde051f7e4586b480fd0dc854581bfd209897437e48ae8e97c5e14f402f31)


```python
# DQN Algorithm部分代码
import parl
import paddle.fluid as fluid
import copy
#from parl import layers

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
```

agent负责和环境做交互，在交互过程中把生成的数据提供给Algorithm来更新模型 (Model)。其中sample和qlearning中sample一样。即不仅有概率取最优的值，还有一定的概率进行探索，随训练的收敛探索程度降低。

![](https://ai-studio-static-online.cdn.bcebos.com/ff2f42ee646e407d8c05fa133f9227d56f7c515f41494e7087d07f79bbc0dfb2)

```python
# DQN Agent部分代码
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
```

此次任务中DQN的训练流程设置：
![](https://ai-studio-static-online.cdn.bcebos.com/46bd30b383a946e5a3b0679d62c401bcb87866883b7d44208118437c9b6ff43c)

```shell
# 训练文件路径在parl/parl_DQN/car_train.py   
# 为notebook运行方便 这里仅将episode设置为10
cd ~
cd parl/parl_DQN
python3 car_train.py
```

训练过程：
![](https://ai-studio-static-online.cdn.bcebos.com/c32a29efc3704451aac3a1e1276f156454af6bba7d5b4693b123f3a21f0b2a74)

#### 2.2 使用parl实现DDPG训练自动泊车

DDPG模型使用单步更新的policy网络，借鉴了DQN的target网络和经验回放，将离散动作扩展到了连续的动作空间。

![](https://ai-studio-static-online.cdn.bcebos.com/d543f7a49b8041bfaf4ddb27747bc853b9b628a2807547b9a9828b6750dae966)
![](https://ai-studio-static-online.cdn.bcebos.com/d050895873f349b3af8f06d248c9a72081e0ba01463e4c75a46903c17fa526e3)

实现代码在parl/parl_DDPG文件夹中。

补充：使用stable_baselines3完成DQN训练：

```shell
#为方便演示同样调小迭代次数
cd ~
pip install stable-baselines3==1.5.0
pip install importlib-metadata==4.0
cd parl/RL_use_torch
python3 DQN.py
```

使用DQN算法针对小车不同的初始位置和不同入库方式进行多次训练。在不同难度的任务中，经过不同时间的训练都能收敛，实现模拟泊车：

![](https://ai-studio-static-online.cdn.bcebos.com/3a394f05376c475cbd5d2eed1444e0f06a2c992b364b4cbfb1633e48596bac89)

![](https://ai-studio-static-online.cdn.bcebos.com/b892617de4234b2b94546057b4f9901ff03577a454b345faa591682096b7247c)

![](https://ai-studio-static-online.cdn.bcebos.com/cab9cc587746434082cb17f8d380c82a088af4d59789456fbfb0f742439a2867)

##  三、实地泊车的实现

使用强化学习训练后的模型模拟泊车完成后，将模拟的参数通过请求发送到小车完成实地停车。现实中汽车的停车场景应该用DDPG训练的模型对应的连续动作（角度和速度）来完成。但用于模拟的小车是以收发离散请求的方式行动，所以在这里使用DQN训练出的模型。通过运行训练完成的模型，可以得到小车每一步的行动参数。但由于收发请求存在延迟且小车舵机反应不出微小的角度变化，可以先计算每隔一段时间的平均参数再发送请求，实现小车实际泊车。
通过强化学习的模型得到运动参数：

```shell
cd ~
cd parl
python3 evaluate_process.py
```

实地泊车演示:

![](https://ai-studio-static-online.cdn.bcebos.com/80373a406f9f433aab0396fa0e5b472c2fd6a873971749bcb8de2cd85cab12d6)

![](https://ai-studio-static-online.cdn.bcebos.com/9034ffb535c04943ab79502c7811ccdd0745f9a02da949c9a6acafd77ba262d9)

![](https://ai-studio-static-online.cdn.bcebos.com/2286a01cf69d4ada8ff95d15ae3d68d248312f7ad708425687261e55b508faaa)

![](https://ai-studio-static-online.cdn.bcebos.com/1e44ed21fb17468d8a8385bdfbf4fb67b509196a56584605b1119d5e9c75e6c0)