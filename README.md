# Tic Tac Toe 井字遊戲
## 專案結構
model_play：執行檔案此開始遊戲

可選擇與AI模型遊玩(先手、後手)或者雙方都是玩家

-----------------------------

model training：為訓練model的執行檔

以Q-learning進行訓練，Q-table shape為(3,3,3,3,3,3,3,3,3,9)共十維

前九個維度代表state，最後一個維度代表action

-----------------------------

model：資料夾中存放訓練好的模型

p1為先手model，p2為後手model，兩者分開訓練

