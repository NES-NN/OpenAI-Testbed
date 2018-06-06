FROM ubuntu:18.04
LABEL maintainer="Joshua Beemster <joshua.a.beemster@gmail.com>"

# 0. Setup environment including timezone.
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

# 0. VINE click library wants utf-8 input
ENV LANG=C.UTF.8

# 1.2 Install tzdata
RUN echo "tzdata tzdata/Areas select Europe" > /tmp/preseed.txt; \
    echo "tzdata tzdata/Zones/Europe select Berlin" >> /tmp/preseed.txt; \
    debconf-set-selections /tmp/preseed.txt && \
    rm -f /etc/timezone && \
    rm -f /etc/localtime 
    #&& \
    #apt-get install -y tzdata

# 1. Installing python2, python3 and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip python-pip python3 python git unzip fceux nano python3-matplotlib tzdata python3-tk

# 1.1 Upgrade pip and pip3
RUN apt-get install -y python3-distutils && pip3 install --upgrade pip setuptools && pip install --upgrade pip


# 2. installing X and xvfb so we can SEE the action using a remote desktop access (VNC)
# and because this is the last apt, let's clean up after ourselves
RUN apt-get install -y x11vnc xvfb fluxbox wmctrl && \
        apt-get clean && \
        rm -rf /var/lib/apt/lists/*

# 3. Install Neat dependencies
RUN pip3 install scipy neat-python==0.8 argparse

# 4. Install Super Mario environment bundle for OpenAI Gym
COPY gym_super_mario /opt/gym_super_mario/
RUN cd /opt/gym_super_mario/ && python3 setup.py install

# 5. Install Uber VINE visualisation tool
RUN pip3 install -U matplotlib
RUN pip install colour
RUN python3 -m pip install --upgrade pandas==0.22.0
RUN cd /opt && \
        git clone https://github.com/uber-common/deep-neuroevolution.git


# VNC Server
EXPOSE 5900

COPY run.sh /
CMD ["/run.sh", "--allow-root"]
