FROM ubuntu:18.04

WORKDIR /home
# if not in China, you can delete the following two lines
COPY ./sources.list.163 ./sources.list.163
RUN mv /etc/apt/sources.list /etc/apt/sources.list.bak && mv sources.list.163 /etc/apt/sources.list

RUN apt-get update && apt-get install -y sudo curl zsh wget git tmux vim build-essential python3-dev python3-pip

RUN pip3 install --upgrade pip

RUN pip3 install --upgrade tensorflow

RUN pip3 install torch torchvision

RUN pip3 install transformers