#参考了19级学长的小车运动模拟及环境搭建

import os
import random
import time

import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces


class CustomEnv(gym.GoalEnv):
    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, render=True, base_path=os.getcwd(), mode='2', manual=False, multi_obs=False, render_video=False):
        """
        初始化环境

        :param render: 是否渲染GUI界面
        :param base_path: 项目路径
        :param car_type: 小车类型（husky）
        :param mode: 任务类型
        :param manual: 是否手动操作
        :param multi_obs: 是否使用多个observation
        :param render_video: 是否渲染视频
        """

        self.base_path = base_path
        self.manual = manual
        self.multi_obs = multi_obs
        self.mode = mode
        assert self.mode in ['1', '2', '3']

        self.car = None
        self.done = False
        self.goal = None
        self.desired_goal = None

        self.ground = None
        self.left_wall1 = None
        self.right_wall1 = None
        self.front_wall1 = None
        self.left_wall2 = None
        self.right_wall2 = None
        self.front_wall2 = None
        self.left_wall3 = None
        self.right_wall3 = None
        self.front_wall3 = None
        self.left_wall4 = None
        self.right_wall4 = None
        self.front_wall4 = None
        self.parked_car1 = None
        self.parked_car2 = None

        # 定义状态空间
        obs_low = np.array([0, 0, -1, -1, -1, -1])
        obs_high = np.array([20, 20, 1, 1, 1, 1])
        if multi_obs:
            self.observation_space = spaces.Dict(
                spaces={
                    "observation": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                    "achieved_goal": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                    "desired_goal": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
                }
            )
        else:
            self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # 定义动作空间
        self.action_space = spaces.Discrete(4)  # 4种动作：前进、后退、左转、右转

        # self.reward_weights = np.array([1, 0.3, 0, 0, 0.02, 0.02])
        self.reward_weights = np.array([0.65, 0.65, 0, 0, 0.1, 0.1])
        self.target_orientation = None
        self.start_orientation = None

        self.action_steps = 5
        self.step_cnt = 0
        self.step_threshold = 500

        if render:
            self.client = p.connect(p.GUI)
            time.sleep(1. / 240.)
        else:
            self.client = p.connect(p.DIRECT)
            time.sleep(1. / 240.)
        if render and render_video:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

    def render(self, mode='rgb_array'):
        """
        渲染当前画面

        :param mode: 渲染模式
        """

        p.stepSimulation(self.client)
        time.sleep(1. / 240.)

    def reset(self):
        """
        重置环境

        """

        p.resetSimulation(self.client)
        p.setGravity(0, 0, -10)

        # 加载地面
        self.ground = p.loadURDF(os.path.join(self.base_path, "env_car_model/arena_new.urdf"), basePosition=[0, 0, 0.005], useFixedBase=10)

        #p.addUserDebugLine([-3.5, -3.5, 0.02], [-3.5, 3.5, 0.02], [0.75, 0.75, 0.75], 5)
        #p.addUserDebugLine([-3.5, -3.5, 0.02], [3.5, -3.5, 0.02], [0.75, 0.75, 0.75], 5)
        #p.addUserDebugLine([3.5, 3.5, 0.02], [3.5, -3.5, 0.02], [0.75, 0.75, 0.75], 5)
        #p.addUserDebugLine([3.5, 3.5, 0.02], [-3.5, 3.5, 0.02], [0.75, 0.75, 0.75], 5)

        """
        # mode = 1  测试环境
        if self.mode == '1':
            self.left_wall1 = p.loadURDF(os.path.join(self.base_path, "env_car_model/up/side_boundary.urdf"), basePosition=[1.3, 2.1, 0.03], useFixedBase=10)
            self.right_wall1 = p.loadURDF(os.path.join(self.base_path, "env_car_model/up/side_boundary.urdf"), basePosition=[2.5, 2.1, 0.03], useFixedBase=10)
            self.front_wall1 = p.loadURDF(os.path.join(self.base_path, "env_car_model/up/front_boundary_ru.urdf"), basePosition=[1.9, 2.8, 0.03], useFixedBase=10)
        else:
            #p.loadURDF(os.path.join(self.base_path, "env_car_model/up/side_boundary.urdf"), basePosition=[1.3, 2.1, 0.03], useFixedBase=10)
            #p.loadURDF(os.path.join(self.base_path, "env_car_model/up/side_boundary.urdf"), basePosition=[2.5, 2.1, 0.03], useFixedBase=10)
            p.loadURDF(os.path.join(self.base_path, "env_car_model/up/front_boundary_ru.urdf"), basePosition=[1.9, 2.8, 0.03], useFixedBase=10)
        p.addUserDebugLine([1.4, 1.5, 0.02], [1.4, 2.7, 0.02], [0.98, 0.98, 0.98], 2.5)
        p.addUserDebugLine([1.4, 1.5, 0.02], [2.4, 1.5, 0.02], [0.98, 0.98, 0.98], 2.5)
        p.addUserDebugLine([2.4, 2.7, 0.02], [1.4, 2.7, 0.02], [0.98, 0.98, 0.98], 2.5)
        p.addUserDebugLine([2.4, 2.7, 0.02], [2.4, 1.5, 0.02], [0.98, 0.98, 0.98], 2.5)
        """
        if self.mode == '2':
            self.left_wall4 = p.loadURDF(os.path.join(self.base_path, "env_car_model/down/side_boundary_rd.urdf"), basePosition=[-0.5, 0.5, -0.50], useFixedBase=10)   #-0.4, 1.15, 0.01
            self.right_wall4 = p.loadURDF(os.path.join(self.base_path, "env_car_model/down/side_boundary_rd.urdf"), basePosition=[0.8, 0.5, -0.50], useFixedBase=10)   #0.9, 1.15, 0.01
            self.front_wall4 = p.loadURDF(os.path.join(self.base_path, "env_car_model/down/front_boundary_rd.urdf"), basePosition=[0.45, -0.2, -0.50], useFixedBase=10)

            self.front_wall5 = p.loadURDF(os.path.join(self.base_path, "env_car_model/down/front_boundary_rd2.urdf"), basePosition=[1.5, 1.4, -0.50], useFixedBase=10) #
            self.front_wall6 = p.loadURDF(os.path.join(self.base_path, "env_car_model/up/side_boundary2.urdf"), basePosition=[2.65, 0.0, -0.50], useFixedBase=10)
            self.front_wall7 = p.loadURDF(os.path.join(self.base_path, "env_car_model/down/front_boundary_rd3.urdf"), basePosition=[0.0, -1.25, -0.50], useFixedBase=10)
            self.front_wall8 = p.loadURDF(os.path.join(self.base_path, "env_car_model/up/side_boundary2.urdf"), basePosition=[-2.65, 0.0, -0.50], useFixedBase=10)
            self.front_wall9 = p.loadURDF(os.path.join(self.base_path, "env_car_model/down/front_boundary_rd5.urdf"), basePosition=[-1.9, 1.4, -0.50], useFixedBase=10) #
            #self.front_wall4 = p.loadURDF(os.path.join(self.base_path, "env_car_model/test/roll.urdf"), basePosition=[0, 0, 1], useFixedBase=10) #0.55, 0.55, 0.01

            p.addUserDebugLine([-0.35, -0.1, 0.02], [1.15, -0.1, 0.02], [0.98, 0.98, 0.98], 2.5)
            p.addUserDebugLine([-1.1, 1.4, 0.02], [-0.35, -0.1, 0.02], [0.98, 0.98, 0.98], 2.5)
            p.addUserDebugLine([1.15,-0.1, 0.02], [0.4, 1.4, 0.02], [0.98, 0.98, 0.98], 2.5)
            ##p.addUserDebugLine([-1.1, 1.4, 0.02], [0.4, 1.4, 0.02], [0.98, 0.98, 0.98], 2.5)

            p.addUserDebugLine([-2.5, 1.4, 0.02], [-1.1, 1.4, 0.02], [0.98, 0.98, 0.98], 2.5)
            p.addUserDebugLine([0.4, 1.4, 0.02], [2.0, 1.4, 0.02], [0.98, 0.98, 0.98], 2.5)

        if self.mode == '3':
            self.left_wall4 = p.loadURDF(os.path.join(self.base_path, "env_car_model/down/side_boundary_rd.urdf"), basePosition=[-0.5, 0.5, -0.50], useFixedBase=10)   #-0.4, 1.15, 0.01
            self.right_wall4 = p.loadURDF(os.path.join(self.base_path, "env_car_model/down/side_boundary_rd.urdf"), basePosition=[0.8, 0.5, -0.50], useFixedBase=10)   #0.9, 1.15, 0.01
            self.front_wall4 = p.loadURDF(os.path.join(self.base_path, "env_car_model/down/front_boundary_rd.urdf"), basePosition=[0.45, -0.2, -0.50], useFixedBase=10)

            self.front_wall5 = p.loadURDF(os.path.join(self.base_path, "env_car_model/down/front_boundary_rd2.urdf"), basePosition=[1.5, 1.4, -0.50], useFixedBase=10)
            self.front_wall6 = p.loadURDF(os.path.join(self.base_path, "env_car_model/up/side_boundary2.urdf"), basePosition=[2.65, 0.0, -0.50], useFixedBase=10)
            self.front_wall7 = p.loadURDF(os.path.join(self.base_path, "env_car_model/down/front_boundary_rd3.urdf"), basePosition=[0.0, -1.25, -0.50], useFixedBase=10)
            self.front_wall8 = p.loadURDF(os.path.join(self.base_path, "env_car_model/up/side_boundary2.urdf"), basePosition=[-2.65, 0.0, -0.50], useFixedBase=10)
            self.front_wall9 = p.loadURDF(os.path.join(self.base_path, "env_car_model/down/front_boundary_rd5.urdf"), basePosition=[-1.9, 1.4, -0.50], useFixedBase=10)
            #self.front_wall4 = p.loadURDF(os.path.join(self.base_path, "env_car_model/test/roll.urdf"), basePosition=[0, 0, 1], useFixedBase=10) #0.55, 0.55, 0.01

            p.addUserDebugLine([-0.35, -0.1, 0.02], [1.15, -0.1, 0.02], [0.98, 0.98, 0.98], 2.5)
            p.addUserDebugLine([-1.1, 1.4, 0.02], [-0.35, -0.1, 0.02], [0.98, 0.98, 0.98], 2.5)
            p.addUserDebugLine([1.15,-0.1, 0.02], [0.4, 1.4, 0.02], [0.98, 0.98, 0.98], 2.5)
            #p.addUserDebugLine([-1.1, 1.4, 0.02], [0.4, 1.4, 0.02], [0.98, 0.98, 0.98], 2.5)

            p.addUserDebugLine([-2.5, 1.4, 0.02], [-1.1, 1.4, 0.02], [0.98, 0.98, 0.98], 2.5)
            p.addUserDebugLine([0.4, 1.4, 0.02], [2.0, 1.4, 0.02], [0.98, 0.98, 0.98], 2.5)
            

        #basePosition = [0.265, 1.025,0]
        #basePosition = [-1.5,2.5, 0.2]
        if self.mode == '1':
            self.goal = np.array([3.8 / 2, 4.2 / 2])
            self.start_orientation = [0, 0, np.pi * 3 / 2]
            self.target_orientation = np.pi * 3 / 2
            basePosition = [1.9, -0.2, 0.2]
        elif self.mode == '2':
            self.goal = np.array([0.265, 0.425])
            self.start_orientation = [0, 0, np.pi * 2 / 2]
            self.target_orientation = 2.070143
            basePosition = [-1.5,2.5, 0.0]   
        elif self.mode == '3':
            self.goal = np.array([0.265, 0.425])
            self.start_orientation = [0, 0, np.pi * 1 / 3]
            self.target_orientation = 2.070143
            #basePosition = [-1.5,2.5, 0.2]         #倒车入库 1226_1383 15
            #basePosition = [0.8,2.0, 0.2]
            #basePosition = [1.5,2.7, 0.2]         #直行1226_2241
            basePosition = [-0.8,3.0, 0.0]

        self.desired_goal = np.array([self.goal[0], self.goal[1], 0.0, 0.0, np.cos(self.target_orientation), np.sin(self.target_orientation)])

        # 加载小车
        self.t = Car(self.client, basePosition=basePosition, baseOrientationEuler=self.start_orientation, action_steps=self.action_steps)
        self.car = self.t.car

        # 获取当前observation
        car_ob, self.vector = self.t.get_observation()
        observation = np.array(list(car_ob))

        self.step_cnt = 0

        if self.multi_obs:
            observation = {
                'observation': observation,
                'achieved_goal': observation,
                'desired_goal': self.desired_goal
            }

        return observation

    def distance_function(self, pos):
        """
        计算小车与目标点的距离（2-范数）

        :param pos: 小车当前坐标 [x, y, z]
        :return: 小车与目标点的距离
        """

        return np.sqrt(pow(pos[0] - self.goal[0], 2) + pow(pos[1] - self.goal[1], 2))

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        计算当前步的奖励

        :param achieved_goal: 小车当前位置 [x, y, z]
        :param desired_goal: 目标点 [x, y, z]
        :param info: 信息
        :return: 奖励
        """

        p_norm = 0.5
        reward = -np.power(np.dot(np.abs(achieved_goal - desired_goal), np.array(self.reward_weights)), p_norm)

        return reward

    def judge_collision(self):
        """
        判断小车与墙壁、停放着的小车是否碰撞

        :return: 是否碰撞
        """

        done = False
        if self.mode == '1':
            points1 = p.getContactPoints(self.car, self.left_wall1)
            points2 = p.getContactPoints(self.car, self.right_wall1)
            points3 = p.getContactPoints(self.car, self.front_wall1)
        elif self.mode == '2':
            points1 = p.getContactPoints(self.car, self.left_wall4)
            points2 = p.getContactPoints(self.car, self.right_wall4)
            points3 = p.getContactPoints(self.car, self.front_wall4)

            points5 = p.getContactPoints(self.car, self.front_wall5)
            points6 = p.getContactPoints(self.car, self.front_wall6)
            points7 = p.getContactPoints(self.car, self.front_wall7)
            points8 = p.getContactPoints(self.car, self.front_wall8)
            points9 = p.getContactPoints(self.car, self.front_wall9)
        elif self.mode == '3':
            points1 = p.getContactPoints(self.car, self.left_wall4)
            points2 = p.getContactPoints(self.car, self.right_wall4)
            points3 = p.getContactPoints(self.car, self.front_wall4)

            points5 = p.getContactPoints(self.car, self.front_wall5)
            points6 = p.getContactPoints(self.car, self.front_wall6)
            points7 = p.getContactPoints(self.car, self.front_wall7)
            points8 = p.getContactPoints(self.car, self.front_wall8)
            points9 = p.getContactPoints(self.car, self.front_wall9)
         
        if len(points1) or len(points2) or len(points3) or len(points5) or len(points6) or len(points7) or len(points8) or len(points9):
            done = True
        return done

    def step(self, action):
        """
        环境步进

        :param action: 小车动作
        :return: observation, reward, done, info
        """

        self.t.apply_action(action)  # 小车执行动作
        p.stepSimulation()
        car_ob, self.vector = self.t.get_observation()  # 获取小车状态

        position = np.array(car_ob[:2])
        distance = self.distance_function(position)
        reward = self.compute_reward(car_ob, self.desired_goal, None)

        if self.manual:
            print(f'dis: {distance}, reward: {reward}, center: {self.goal}, pos: {car_ob}')

        self.done = False
        self.success = False

        if distance < 0.1:
            self.success = True
            self.done = True

        self.step_cnt += 1
        if self.step_cnt > self.step_threshold:  # 限制episode长度为step_threshold
            self.done = True
        if car_ob[2] < -2:  # 小车掉出环境
            # print('done! out')
            reward = -500
            self.done = True
        if self.judge_collision():  # 碰撞
            # print('done! collision')
            reward = -500
            self.done = True
        if self.done:
            self.step_cnt = 0

        observation = np.array(list(car_ob))
        if self.multi_obs:
            observation = {
                'observation': observation,
                'achieved_goal': observation,
                'desired_goal': self.desired_goal
            }

        info = {'is_success': self.success}

        return observation, reward, self.done, info

    def seed(self, seed=None):
        """
        设置环境种子

        :param seed: 种子
        :return: [seed]
        """

        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    def close(self):
        """
        关闭环境

        """

        p.disconnect(self.client)


class Car:
    def __init__(self, client, basePosition=[0, 0, 0.2], baseOrientationEuler=[0, 0, np.pi / 2],
                 max_velocity=3, max_force=50, action_steps=None):
        """
        初始化小车

        :param client: pybullet client
        :param basePosition: 小车初始位置
        :param baseOrientationEuler: 小车初始方向

        """

        self.client = client

        #urdfname = "env_car_model/car1.urdf"
        urdfname = "env_car_model/car_model2/car_model2.urdf"
        self.car = p.loadURDF(fileName=urdfname, basePosition=basePosition, baseOrientation=p.getQuaternionFromEuler(baseOrientationEuler))

        self.steering_joints = [0, 2]
        self.drive_joints = [1, 3, 4, 5]

        self.max_velocity = max_velocity
        self.max_force = max_force
        self.action_steps = action_steps

    def apply_action(self, action):
        """
        小车执行动作

        :param action: 动作
        """

        velocity = self.max_velocity
        force = self.max_force

        if action == 0:  # 前进
            for i in range(self.action_steps):
                for joint in range(2, 6):
                    p.setJointMotorControl2(self.car, joint, p.VELOCITY_CONTROL, targetVelocity=velocity, force=force)
                p.stepSimulation()
        elif action == 1:  # 后退
            for i in range(self.action_steps):
                for joint in range(2, 6):
                    p.setJointMotorControl2(self.car, joint, p.VELOCITY_CONTROL, targetVelocity=-velocity, force=force)
                p.stepSimulation()
        elif action == 2:  # 左转
            targetVel = 3
            for i in range(self.action_steps):
                for joint in range(2, 6):
                    for joint in range(1, 3):
                        p.setJointMotorControl2(self.car, 2 * joint + 1, p.VELOCITY_CONTROL,
                                                targetVelocity=targetVel, force=force)
                    for joint in range(1, 3):
                        p.setJointMotorControl2(self.car, 2 * joint, p.VELOCITY_CONTROL, targetVelocity=-targetVel,
                                                force=force)
                    p.stepSimulation()
        elif action == 3:  # 右转
            targetVel = 3
            for i in range(self.action_steps):
                for joint in range(2, 6):
                    for joint in range(1, 3):
                        p.setJointMotorControl2(self.car, 2 * joint, p.VELOCITY_CONTROL, targetVelocity=targetVel,
                                                force=force)
                    for joint in range(1, 3):
                        p.setJointMotorControl2(self.car, 2 * joint + 1, p.VELOCITY_CONTROL,
                                                targetVelocity=-targetVel, force=force)
                    p.stepSimulation()
        elif action == 4:  # 停止
            targetVel = 0
            for joint in range(2, 6):
                p.setJointMotorControl2(self.car, joint, p.VELOCITY_CONTROL, targetVelocity=targetVel,
                                        force=force)
            p.stepSimulation()
        else:
            raise ValueError

    def get_observation(self):
        """
        获取小车当前状态

        :return: observation, vector
        """

        position, angle = p.getBasePositionAndOrientation(self.car)  # 获取小车位姿
        angle = p.getEulerFromQuaternion(angle)
        velocity = p.getBaseVelocity(self.car)[0]

        position = [position[0], position[1]]
        velocity = [velocity[0], velocity[1]]
        orientation = [np.cos(angle[2]), np.sin(angle[2])]
        vector = angle[2]

        observation = np.array(position + velocity + orientation)  # 拼接坐标、速度、角度

        return observation, vector
