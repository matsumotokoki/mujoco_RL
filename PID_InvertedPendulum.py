import gym
import math
import numpy as np
import sys
import time

env = gym.make('InvertedPendulum-v2')
Render_Flag = True
max_step = 0
for i in range(20000):
    env.reset()
    Kp = 6.0
    Ki = 0.04
    Kd = 0.03
    goal = 0.0
    e = 0.0
    e1 = 0.0
    e2 = 0.0
    action = -0.01
    action_old = 0.0

    for t in range(200):
        if Render_Flag:
            env.render()
        obs, reward, done, info = env.step(-action)
        action_old = action
        e2 = e1
        e1 = e
        e = goal - obs[1] 
        action = action_old + Kp * (e-e1) + Ki * e + Kd * ((e-e1)-(e1-e2))
        if done or t >= 199:
            if max_step < t:
                max_step = t
            print("episode:%5d, step:%3d, max:%3d"%(i+1,t+1,max_step+1))
            # sys.stdout.write("\repisode:%5d, step:%3d, max:%3d"%(i+1,t+1,max_step+1))
            # sys.stdout.flush()
            # time.sleep(0.1)
            break


