### First time setup
- Put the train.py in the coding challenge's top level folder in order to run it

### To do
- bug fixes:
    - [DONE] monitoring reward (check monitoring wrapper)
    - [DONE] freezing progress (decrease the max round count)
    - other snake body not accounted for in action mask (implement mask in debugging)

- general:
    - implement suicide
    - try the 0-1 bitwise state where a seperate grid is used for eacch valu	wae in the observation
    - try rewarding the best bot's maneuvres

- action:
    - [DONE] check inference on masked model

- state:
    - options to balance size/information:
        - give each block of the snakes a unique identifier 
        (otherwise state does not 'see' how snake is folded)
        - only head and tail
        (might enable escaping dense area by following opponent tail)
        - use simple state (see online snake example)
        - use CNN to condense the state information

- reward:
    - [DONE] penalize not getting candy
    - reward direct progress towards candy
        - enable tracking of one candy (now closest -> might cause issues)
    - a-star progress reward (make progress on optimal path to candy considering obstacles)
    - minimize the maximum risk -> minimax
    - reward based on length difference between bots -> goal is to grow twice as long

