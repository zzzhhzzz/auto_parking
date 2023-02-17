import time
import numpy as np
import gym
import parking_env
from stable_baselines3 import DQN   #使用stable_baselines3训练出的模型
#from parl import DQN       #要切换parl训练出的模型
import pybullet as p

#cloudpickle 需要2.2.0版本


model_path = f'model/model_for_python3.7.zip'  #mode2
#model_path = f'model/mode2/1500000_steps.zip'  #mode2
#model_path = f'model/mode3/3600000_steps.zip'     #mode3

cameraYaw = 0

env = gym.make("parking_env-v0", render=False, mode="2", render_video=False)
#env = gym.make("parking_env-v0", render=True, mode="3", render_video=True)
obs = env.reset()
model = DQN.load(model_path, env=env, print_system_info=True)

p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

episode_return = 0
for i in range(15):

    """
    通过训练好的模型参数计算请求发送数值  结合http_request模板发送请求 建议先简单计算模拟小车与实际小车的速度比
    """
    action, _ = model.predict(obs, deterministic=True)
    if action == 1:
        print("action:倒车")
    elif action == 2:
        print("action:左转")
    elif action == 3:
        print("action:右转")
    elif action == 4:
        print("action:停止")
    obs, reward, done, info = env.step(action)

    position, _ = p.getBasePositionAndOrientation(env.car)
    print(f"小车x坐标：{position[0]}   小车y坐标：{position[1]}")

    position, angle = p.getBasePositionAndOrientation(env.car)
    angle = p.getEulerFromQuaternion(angle)
    a = 180 - angle[2]*57.3
    print(f"小车整体偏角：{a}度")
    velocity = p.getBaseVelocity(env.car)[0]
    print(f"小车x方向速度：{velocity[0]}  小车y方向速度{velocity[1]}")
    a2 = 180-np.arctan(velocity[1]/velocity[0])*57.3
    print(f"小车行进方向：{a2}度")
    angel_final = 270-a2+a
    print(f"(根据转角与偏角计算)http请求应发送angle ：{angel_final}度")
    print("-------------------------------------------------------")

    p.resetDebugVisualizerCamera(
        cameraDistance=4,
        cameraYaw=cameraYaw,
        cameraPitch=-40,
        cameraTargetPosition=position
    )
    time.sleep(20 / 24)

    episode_return += reward
    if done:
        break


env.close()

