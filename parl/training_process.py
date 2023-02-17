import time
import gym
import parking_env
from stable_baselines3 import DQN   #使用stable_baselines3训练出的模型
#from parl.algorithms import DQN    #要切换parl训练出的模型
import pybullet as p


cameraYaw = 0

for step in range(200000, 36000000, 200000):
    model_path = f'model/mode3/{step}_steps.zip'
    env = gym.make("parking_env-v0", render=True, mode="3", render_video=True)  
    obs = env.reset()
    model = DQN.load(model_path, env=env)

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    p.resetDebugVisualizerCamera(
        cameraDistance=4,
        cameraYaw=cameraYaw,
        cameraPitch=-45,
        cameraTargetPosition=[0, 0, 0]
    )
    p.addUserDebugText(
        text=f"DQN train episode : {step}",
        textPosition=[-1.5, 0, 2],
        textColorRGB=[0, 0, 0],
        textSize=2.5
    )

    episode_return = 0
    for i in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        position, _ = p.getBasePositionAndOrientation(env.car)
        episode_return += reward
        if done:
            break
        time.sleep(1 / 500)   


    env.close()
    print(f'step: {step}, episode return: {episode_return}')

