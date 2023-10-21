### First time setup
- Add the path to the snakes module of the coding-challenge-snakes to your pythonpath
to prevent importing errors

### To do
- bug fixes
    - [DONE] monitoring reward (check monitoring wrapper)
    - [DONE] freezing progress (decrease the max round count)

- implement action masking
    - [DONE] check inference on masked model

- make state unique 
    - embed the snake shape into the state

- reward shaping
    - [DONE] penalize not getting candy
    - reward direct progress towards candy
    - a-star progress reward (make progress on optimal path to candy considering obstacles)
    - minimize the maximum risk -> minimax 

- enable randomly sampling a bot during training