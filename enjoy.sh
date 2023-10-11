#!/usr/bin/env sh
(
	cd ../../..
  python -m rl_zoo3.enjoy --algo ppo --gym-packages snakes.bots.brammmieee.env --env snakes -f logs/
)
