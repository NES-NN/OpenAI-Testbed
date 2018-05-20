#!/bin/bash
set -e

pwd=$(pwd)

docker run \
	--name openai_smb \
	-e VNC_SERVER_PASSWORD=password \
	-p 5900:5900 \
	-v ${pwd}/train:/opt/train \
	-d openai/smb
