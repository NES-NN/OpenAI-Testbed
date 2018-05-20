FROM ubuntu:18.04
LABEL maintainer="Joshua Beemster <joshua.a.beemster@gmail.com>"

# 1. Installing python2, python3 and dependencies
RUN apt-get update && apt install -y --no-install-recommends python3-pip python-pip python3 python git unzip fceux nano

# 1.1 Uppgrade pip and pip3
RUN pip3 install --upgrade pip setuptools && pip install --upgrade pip

# 2. Install OpenAI Gym dependencies
RUN pip3 install gym-pull && pip3 install gym==0.8.2

# 2.1 Install OpenAI-Neat dependencies
RUN pip3 install scipy neat-python==0.8 argparse

# 2.2 Install Super Mario environment bundle for OpenAI Gym
RUN pip3 install ppaquette-gym-super-mario

# 3. installing X and xvfb so we can SEE the action using a remote desktop access (VNC)
# and because this is the last apt, let's clean up after ourselves
RUN apt-get install -y x11vnc xvfb fluxbox wmctrl && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# VNC Server
EXPOSE 5900

COPY run.sh /
CMD ["/run.sh", "--allow-root"]
