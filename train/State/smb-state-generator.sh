#!/bin/bash

export DISPLAY=:1
export SMB_STATE_SAVE_DIR=/opt/train/State/SMB_STATE

/opt/gym_super_mario/fceux/fceux \
	--loadlua /opt/train/State/smb-state-generator.lua \
	/opt/gym_super_mario/ppaquette_gym_super_mario/roms/super-mario.nes
