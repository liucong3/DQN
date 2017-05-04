# DQN
This is an implementation of the *deep reinforcement learning* (DQN) based on
* [Tensorflow](https://www.tensorflow.org)
* [GYM](https://github.com/openai/gym)
## Install dependencies
* Kivy
```bash
sudo add-apt-repository ppa:kivy-team/kivy
sudo apt-get update
sudo apt-get install python-kivy
```
* Tensorflow
```bash
sudo apt-get install python-dev
curl "https://bootstrap.pypa.io/get-pip.py" -o "get-pip.py"
sudo python get-pip.py
sudo pip install tensorflow
```
* GYM
```bash
sudo apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl swig
sudo pip install 'gym[all]'
```
## Play games with UI
```bash
python ui.py --env Breakout
python ui.py --env 0
python ui.py --env 48
```
## Train model
```bash
python learn.py --env Breakout
python learn.py --env Breakout --savePath my_save_path
python learn.py --env 0
python learn.py --env 48
python fast.py --env Breakout
python fast2.py --env Breakout
```
## Game UI with DQN control
```bash
python ui.py --env Breakout --savePath my_save_path/model
```

