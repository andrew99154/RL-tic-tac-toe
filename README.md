Tic Tac Toe 井字遊戲！

執行 model_play 檔案開始遊戲
可選擇與AI模型遊玩(先手、後手)或者雙方都是玩家

model training為訓練model的執行檔
以Q-learning進行訓練，Q-table shape為(3,3,3,3,3,3,3,3,3,9)共十維
前九個維度代表state，最後一個維度代表action

model/
p1為先手model，p2為後手model，兩者分開訓練

歡迎挑戰model找弱點，找到可以幫忙自己練model發PR，tks!