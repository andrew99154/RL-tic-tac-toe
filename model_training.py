import os
import numpy as np
from collections import deque
import random
from matplotlib import pyplot as plt
class Env:
    def __init__(self) -> None:
        self.states = [0]*9
        self.circle_turn = True
        self.game_end = False
        self.moves_amount = 0
        self.winner = 0 # 0=nobody, 1=O, 2=X

    def get_state(self):
        return tuple(self.states)

    def reset(self):
        self.states = [0]*9
        self.circle_turn = True
        self.game_end = False
        self.moves_amount = 0
        self.winner = 0
        return tuple(self.states)

    def now_player(self):
        return 1 if self.circle_turn else 2
    
    def move(self, position):
        if self.circle_turn:
            self.states[position] = 1
        else:
            self.states[position] = 2
        self.moves_amount += 1
        self.circle_turn = not self.circle_turn
        done = self.check_game_end()
        reward = self.cal_reward()
        return tuple(self.states), reward, done 

    def legal_actions(self):
        legal = []
        for i in range(9):
            if self.states[i] == 0:
                legal.append(i)
        return legal

    def check_move_legal(self, position):
        return self.states[position] == 0

    def get_winner(self):
        return self.winner

    def cal_reward(self):
        if self.winner == 1:
            reward = 5
        elif self.winner ==2:
            reward = -7
        else:
            reward = 0
        return reward
    
    def check_game_end(self):
        for i in range(3):
            if self.states[3*i]!=0 and (self.states[3*i] == self.states[3*i+1] and self.states[3*i+1] == self.states[3*i+2]):
                self.game_end = True
                self.winner = self.states[3*i]
            elif self.states[i]!=0 and (self.states[i] == self.states[i+3] and self.states[i] == self.states[i+6]):
                self.game_end = True
                self.winner = self.states[i]
        if not self.game_end:
            if self.states[4]!=0 and ((self.states[0]==self.states[4] and self.states[4]==self.states[8]) or (self.states[2]==self.states[4] and self.states[4]==self.states[6])):
                self.game_end = True
                self.winner = self.states[4]

        if self.moves_amount == 9:
            self.game_end = True
        return self.game_end
    
class Agent:
    def __init__(self,n_episode,s_dim,a_dim,lr,gamma) -> None:
        self.q_table = np.zeros([3]*s_dim+[a_dim],dtype=np.float32)
        self.total_rewards = np.zeros(n_episode+1, dtype=np.float32)
        self.lr = lr
        self.gamma = gamma
        self.buffer_size = 512
        self.mb_size = 4
        self.replay_buffer = deque(maxlen=self.buffer_size)

    def update_qtable(self,state, action, reward, next_state,legal_action):
        curr_q = self.q_table[state+(action,)]
        if next_state != None:
            max_next_q = max(self.q_table[next_state][i] for i in legal_action)
            self.q_table[state+(action,)] = curr_q + self.lr*(reward + self.gamma*max_next_q - curr_q)
        else:
            self.q_table[state+(action,)] = curr_q + self.lr*(reward - curr_q)
        
        # extend replay buffer
        # if self.mb_size < len(self.replay_buffer): return
        # for i in np.random.choice(len(self.replay_buffer), self.mb_size, replace=False):
        #     state, action, reward, next_state = self.replay_buffer[i]
        #     curr_q = self.q_table[state+(action,)]
        #     if next_state != None:
        #         max_next_q = max(self.q_table[next_state][i] for i in legal_action)
        #         self.q_table[state+(action,)] = curr_q + self.lr*(reward + self.gamma*max_next_q - curr_q)
        #     else:
        #         self.q_table[state+(action,)] = curr_q + self.lr*(reward - curr_q)

    def put_in_buffer(self,state, action, reward, next_state):
        self.replay_buffer.append((state,action,reward,next_state))

    def choose_action(self,legal_a,eps,state):
        if np.random.random() > eps:
            q_values = []
            for i in range(9):
                if i in legal_a:
                    q_values.append(0 if np.isnan(self.q_table[state][i]) else self.q_table[state][i])
                else:
                    self.q_table[state][i] = np.NINF
            max_idxs = np.where(q_values==np.max(q_values))[0]
            if len(max_idxs)==0:
                print(state)
                print(q_values)
            else:
                action_idx = random.choice(max_idxs)
                action = legal_a[action_idx]
        else:
            action = random.choice(legal_a)
        return action
    
    def save_model(self,name):
        current_dir = os.path.dirname(__file__)
        new_path = os.path.join(current_dir, "model","qtable"+name+".npy")
        with open(new_path,"wb") as f:
            print("saving the Q-table...", end = "")
            np.save(f,self.q_table)
            print("Done!")

def main():
    #Q-learning args
    n_episode = 10000
    n_disp = 1000 #幾步印一次學習狀況
    lr = 0.3 #learning rate
    gamma = 0.9 #discount factor
    eps = 1.0 #epsilon greedy
    s_dim = 9
    a_dim = 9
    total_rewards = np.zeros(n_episode+1, dtype=np.float32) #紀錄所有episode的reward變化

    first_step = True
    env = Env()
    player = ["_","O","X"]
    p1 = Agent(n_episode,s_dim,a_dim,lr,gamma)
    p2 = Agent(n_episode,s_dim,a_dim,lr,gamma)

    for i_episode in range(n_episode):
        p1_state = env.reset()

        while True:
            #agent turn
            legal_action = env.legal_actions()
            p1_action = p1.choose_action(legal_action,eps,p1_state)
            p2_next_state, reward, done = env.move(p1_action)
            if done:
                break

            if not first_step:
                p2.update_qtable(p2_state,p2_action,reward*-1,p2_next_state,legal_action)
            else:
                first_step = False
            p2_state = p2_next_state

            #player turn
            legal_action = env.legal_actions()
            p2_action = p2.choose_action(legal_action,eps,p2_state)
            p1_next_state, reward, done = env.move(p2_action)
            if done:
                break

            legal_action = env.legal_actions()
            p1.update_qtable(p1_state,p1_action,reward,p1_next_state,legal_action)
            p1_state = p1_next_state

        p1.update_qtable(p1_state,p1_action,reward,None,None)
        p2.update_qtable(p2_state,p2_action,reward*-1,None,None)
        total_rewards[i_episode] = reward
        
        eps = np.exp(-0.2 - 4.0*i_episode / n_episode)

        if i_episode % n_disp ==0:
            print("[{:5d}/{:5d}] total_reward = {:.3f}, avg_total_reward = {:.3f}, epsilon = {:.3f}".format(
                i_episode, n_episode,reward, total_rewards[:i_episode+1].mean(), eps
            ))
            
    p1.save_model("-p1")
    p2.save_model("-p2")

    window_size = 50
    episodes = np.arange(0,n_episode)
    mean_total_rewards = np.zeros(n_episode)
    std_total_rewards = np.zeros(n_episode)
    for i in range(n_episode):
        left = max(0,i-window_size)
        right = min(i+window_size,len(total_rewards))
        mean_total_rewards[i] = total_rewards[left:right].mean()
        std_total_rewards[i] = total_rewards[left:right].std()

    plt.xlim(0,n_episode)
    plt.ylim(0,5)
    plt.grid()
    plt.plot(episodes,mean_total_rewards, color = "red")
    plt.fill_between(
        episodes,
        mean_total_rewards+std_total_rewards,
        mean_total_rewards-std_total_rewards,
        color = "red",
        alpha = 0.4
    )
    plt.show()            

if __name__ == "__main__":
    main()