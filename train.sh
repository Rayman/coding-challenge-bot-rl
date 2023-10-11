#!/usr/bin/env sh
(
	cd ../../..
	python -m rl_zoo3.train --algo ppo --gym-packages snakes.bots.brammmieee.env --env snakes -conf snakes/bots/brammmieee/ppo.yml
)