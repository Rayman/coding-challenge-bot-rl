### First time setup
- Add the path to the snakes module of the coding-challenge-snakes to your pythonpath
to prevent importing errors

### TODO
- bug fixes
    -> monitoring reward (check monitoring wrapper)
    -> freezing progress (decrease the max round count)

- implement action masking
    -> check inference on masked model

- make state unique 
    -> embed the snake shape into the state

- reward shaping
    -> penalize not getting candy
    -> reward direct progress towards candy
    -> a-star progress reward (make progress on optimal path to candy considering obstacles)

- enable randomly sampling a bot for a random amount of games