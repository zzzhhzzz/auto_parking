import numpy
import pybullet as p
import gym
import parking_env


env = gym.make("parking_env-v0", render=True, manual=True, mode="2")    #mode = 2 or 3  
env.reset()

done = False
while True:
    keys = p.getKeyboardEvents()
    for k, v in keys.items():
        if k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN):
            action = 2
        if k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN):
            action = 3
        if k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN):
            action = 0
        if k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN):
            action = 1
        if (k == p.B3G_LEFT_ARROW or k == p.B3G_RIGHT_ARROW or k == p.B3G_UP_ARROW or k == p.B3G_DOWN_ARROW) and (v & p.KEY_WAS_RELEASED):
            action = 4
        else:
            pass
        next_state, reward, done, _ = env.step(action)


