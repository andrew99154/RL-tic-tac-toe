import os
class Env:
    def __init__(self) -> None:
        self.states = [2]*9
        self.circle_turn = True
        self.game_end = False
        self.moves_amount = 0
        self.winner = 2 # 0=X, 1=O, 2=nobody

    def get_state(self):
        return self.states
    
    def now_player(self):
        return self.circle_turn

    def isEnd(self):
        return self.game_end
    
    def move(self, position):
        if self.circle_turn:
            self.states[position] = 1
        else:
            self.states[position] = 0
        self.moves_amount += 1
        self.circle_turn = not self.circle_turn
        self.check_game_end()

    def check_move_legal(self, position):
        return self.states[position] == 2

    def get_winner(self):
        return self.winner

    def check_game_end(self):
        for i in range(3):
            if self.states[3*i]!=2 and (self.states[3*i] == self.states[3*i+1] and self.states[3*i+1] == self.states[3*i+2]):
                self.game_end = True
                self.winner = self.states[3*i]
            elif self.states[i]!=2 and (self.states[i] == self.states[i+3] and self.states[i] == self.states[i+6]):
                self.game_end = True
                self.winner = self.states[i]
        if not self.game_end:
            if self.states[4]!=2 and ((self.states[0]==self.states[4] and self.states[4]==self.states[8]) or (self.states[2]==self.states[4] and self.states[4]==self.states[6])):
                self.game_end = True
                self.winner = self.states[4]

        if self.moves_amount == 9:
            self.game_end = True

            
def print_state(player, game_state):
    os.system('cls' if os.name == 'nt' else 'clear')
    for i in range(0,3):
        print(f"{player[game_state[3*i]]:>2}{player[game_state[3*i+1]]:>2}{player[game_state[3*i+2]]:>2}")


def main():
    game_env = Env()
    player = ["X","O","_"]
    while not game_env.isEnd():
        print_state(player,game_env.get_state())
        input_legal = False        
        while not input_legal:
            player_action = input(f"{player[game_env.now_player()]} 的回合，請輸入數字1~9(左上至右下編號)：")
            if not player_action.isnumeric():
                print("格式錯誤，請輸入數字！")
                continue
            player_action = int(player_action)-1
            if player_action<0 or player_action>8:
                print("超過數字範圍，請輸入正確數字！")
            elif not game_env.check_move_legal(player_action):
                print("這格已經被畫過了，請選擇其他格！")
            else:
                input_legal = True
                game_env.move(player_action)

        if game_env.isEnd():
            print_state(player,game_env.get_state())

            winner = game_env.get_winner()
            if winner == 2:
                print("平手，遊戲結束！")
            else:
                print(f"贏家是{player[winner]}，遊戲結束！")

            

    

if __name__ == "__main__":
    main()