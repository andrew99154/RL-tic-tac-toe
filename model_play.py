import os
import numpy as np
class Env:
    def __init__(self) -> None:
        self.states = [0]*9
        self.circle_turn = True
        self.game_end = False
        self.moves_amount = 0
        self.winner = 0 # 0=X, 1=O, 2=nobody

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
            reward = 300
        elif self.winner ==2:
            reward = -400
        else:
            reward = -1
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

            
def print_state(player, game_state):
    os.system('cls' if os.name == 'nt' else 'clear')
    for i in range(0,3):
        print(f"{player[game_state[3*i]]:>2}{player[game_state[3*i+1]]:>2}{player[game_state[3*i+2]]:>2}")
    print("\n")

def play_local():
    env = Env()
    player = ["_","O","X"]

    while True:
        print_state(player,env.get_state())
        if env.circle_turn:
            input_legal = False        
            while not input_legal:
                player_action = input(f"{player[env.now_player()]} 的回合，請輸入數字1~9(左上至右下編號)：")
                if not player_action.isnumeric():
                    print("格式錯誤，請輸入數字！")
                    continue
                player_action = int(player_action)-1
                if player_action<0 or player_action>8:
                    print("超過數字範圍，請輸入正確數字！")
                elif not env.check_move_legal(player_action):
                    print("這格已經被畫過了，請選擇其他格！")
                else:
                    input_legal = True
                    _, _, done = env.move(player_action)
        else: 
            input_legal = False        
            while not input_legal:
                player_action = input(f"{player[env.now_player()]} 的回合，請輸入數字1~9(左上至右下編號)：")
                if not player_action.isnumeric():
                    print("格式錯誤，請輸入數字！")
                    continue
                player_action = int(player_action)-1
                if player_action<0 or player_action>8:
                    print("超過數字範圍，請輸入正確數字！")
                elif not env.check_move_legal(player_action):
                    print("這格已經被畫過了，請選擇其他格！")
                else:
                    input_legal = True
                    _, _, done = env.move(player_action)

        if done:
            print_state(player,env.get_state())
            winner = env.get_winner()
            if winner == 0:
                print("平手，遊戲結束！")
            else:
                print(f"贏家是{player[winner]}，遊戲結束！")
            break

def play_with_agent(player_first):

    env = Env()
    player = ["_","O","X"]
    state = env.reset()
    current_dir = os.path.dirname(__file__)
    if player_first:
        new_path = os.path.join(current_dir, "model","qtable-p2-strong.npy")
        with open(new_path,"rb") as f:
            q_table = np.load(f)
    else:
        new_path = os.path.join(current_dir, "model","qtable-p1-strong.npy")
        with open(new_path,"rb") as f:
            q_table = np.load(f)

    if player_first:
        print_state(player,env.get_state())
        while True:
            p2_action = input(f"{player[env.now_player()]} 的回合，請輸入數字1~9(左上至右下編號)：")
            if not p2_action.isnumeric():
                print("格式錯誤，請輸入數字！")
                continue
            p2_action = int(p2_action)-1
            if p2_action<0 or p2_action>8:
                print("超過數字範圍，請輸入正確數字！")
            elif not env.check_move_legal(p2_action):
                print("這格已經被畫過了，請選擇其他格！")
            else:
                break
        state, _, done = env.move(p2_action)

    while True:
        print_state(player,env.get_state())
        #目前只訓練ai玩先手(圈圈部分)
        #agent turn
        p1_action = np.argmax(q_table[state])
        state, _, done = env.move(p1_action)
        print_state(player,env.get_state())
        if done:
            break

        #player turn
        while True:
            p2_action = input(f"{player[env.now_player()]} 的回合，請輸入數字1~9(左上至右下編號)：")
            if not p2_action.isnumeric():
                print("格式錯誤，請輸入數字！")
                continue
            p2_action = int(p2_action)-1
            if p2_action<0 or p2_action>8:
                print("超過數字範圍，請輸入正確數字！")
            elif not env.check_move_legal(p2_action):
                print("這格已經被畫過了，請選擇其他格！")
            else:
                break
        state, _, done = env.move(p2_action)
        print_state(player,env.get_state())
        if done:
            break

    print_state(player,env.get_state())
    winner = env.get_winner()
    if winner == 0:
        print("平手，遊戲結束！")
    else:
        print(f"贏家是{player[winner]}，遊戲結束！")


def main():
    print("1:與AI遊玩, 2:雙人模式")
    game_mode = ""
    while True:
        game_mode = input("請輸入數字選擇遊戲模式：")
        if game_mode == "1" or game_mode == "2":
            break
        print("輸入錯誤！")

    if game_mode == "1":
        print("1:先手, 2:後手")
        while True:
            first_player = input("請輸入數字選擇遊戲模式：")
            if first_player == "1" or first_player == "2":
                break
            print("輸入錯誤！")

        if first_player=="1":
            while True:
                play_with_agent(True)
                if input("再來一局？(輸入y再來一局或任意鍵退出)") != "y":
                    break
        else:
            while True:
                play_with_agent(False)
                if input("再來一局？(輸入y再來一局或任意鍵退出)") != "y":
                    break
    else:
        while True:
            play_local()
            if input("再來一局？(輸入y再來一局或任意鍵退出)") != "y":
                break
    print("遊戲結束，謝謝遊玩")


if __name__ == "__main__":
    main()