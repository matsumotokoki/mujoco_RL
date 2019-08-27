import gym
import numpy as np

class Q_learning(object):
    def __init__(self):
        self.env = gym.make('InvertedPendulum-v2')
        self.digitalied_num = 10
        self.steps = 200
        self.episodes = 10000000
        self.goal_ave = -150
        self.moving_ave_num = 10
        self.first_prob = 0.75
        self.moving_ave = np.full(self.moving_ave_num,-200)
        self.q_table = np.random.uniform(low=-1,high=1,size=(self.digitalied_num\
                                                        **self.env.observation_space.shape[0],7))
        self.reward_of_episode = 0
        self.render_flag = False
        self.learning_finish = False
        self.alpha = 0.8
        self.gamma = 0.99
        self.bin_pram = []
        pram_low = [-2.4,-3.0,-0.5,-2.0]
        pram_high = [2.4,3.0,0.5,2.0]
        for i in range(self.env.observation_space.shape[0]):
            self.bin_pram.append(np.linspace(pram_low[i],pram_high[i],self.digitalied_num)[1:-1])
            #self.bin_pram.append(np.linspace(self.env.observation_space.low[i],self.env.observation_space.high[i],self.digitalied_num)[1:-1])

    def digitalie(self,obs):
        state = 0
        for i in range(self.env.observation_space.shape[0]):
            state = np.digitize(obs[i],self.bin_pram[i]) * (self.digitalied_num ** i)
        return state
    
    def decide_action(self,next_state,episode):
        epsilon = self.first_prob * (1/(episode+1))
        if epsilon <= np.random.uniform(0,1):
            next_action = np.argmax(self.q_table[next_state])
        else:
            next_action = int(round(self.env.action_space.sample()[0]))
        return next_action

    def update_Q_table(self,next_state,state,action,reward,q_table):
        next_max_q = max(q_table[next_state])
        q_table[state,action] = (1 - self.alpha) * (q_table[state,action]) + \
                                    self.alpha * (reward + self.gamma *next_max_q)
        return q_table

    def run(self):
        max_step = 0
        for episode in range(self.episodes):
            obs = self.env.reset()
            state = self.digitalie(obs)
            action = np.argmax(self.q_table[state])
            self.reward_of_episode = 0

            for i in range(self.steps):
                if self.render_flag or self.learning_finish:
                    self.env.render()
                
                observation ,reward, done, info = self.env.step(action-3)
                if done:
                    if i < 195:
                        reward = -200
                    else:
                        reward = 1
                else:
                    reward = 1

                self.reward_of_episode += reward
                
                next_state = self.digitalie(obs)
                self.q_table = self.update_Q_table(next_state,state,action,reward,self.q_table)
                action = self.decide_action(next_state,episode)
                state = next_state
                

                if done:
                    if max_step < i:
                        max_step = i
                    self.moving_ave = np.hstack((self.moving_ave[1:],self.reward_of_episode))
                    print("----------------")
                    print("episode: {}".format(episode+1))
                    print("reward : {}".format(self.reward_of_episode))
                    print("step   : {}".format(i+1))
                    print("maxstep: {}".format(max_step+1))
                    print("average: {}".format(self.moving_ave.mean()))
                    print("----------------")
                    if self.learning_finish:
                        self.render_flag = True
                    break

            if self.moving_ave.mean() >= self.goal_ave:
                print("Learning is finished!!")
                print("episode: {}".format(episode+1))
                self.learning_finish = True


pendulum = Q_learning()
pendulum.run()
