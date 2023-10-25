### First time setup
- Put the train.py in the coding challenge's top level folder in order to run it

### To do
- bug fixes
    - monitoring reward (check monitoring wrapper)
    - freezing progress (decrease the max round count)
    - other snake body not masked
    - action masking still allows turning in place (see debugging)

- implement action masking
    - [DONE] check inference on masked model

- make state unique 
    - embed the snake shape into the state

- reduce the action space
    - compute locally valid actions (left, right, straight)

- reward shaping
    - [DONE] penalize not getting candy
    - reward direct progress towards candy
    - a-star progress reward (make progress on optimal path to candy considering obstacles)
    - minimize the maximum risk -> minimax 

- [DONE] enable randomly sampling a bot during training

- try to limit the state space
    - checkout png compression technique for lossless compression
    - checkout the local state representation (con: information loss)

