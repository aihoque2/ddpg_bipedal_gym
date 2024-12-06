# ddpg_bipedal_gym
This is an implementation of Deep-Deterministic Policy Gradient (DDPG) in PyTorch that is used to train the bipedal walker in the gym environment.
The agent uses a Policy that is a neural network trained with respect to a critic who predicts the quality of state action pairs
more can be read here: 

https://arxiv.org/pdf/1509.02971.pdf



## to run:
first clone the repo with 
`git clone https://github.com/aihoque2/ddpg_bipedal_gym.git`

then cd into the repository and run

`python ./main.py` 

to recreate the findings I had with my model

## results
see gif:
![](gifs/bipedal.gif)

my agent was able to complete the bipdal course 3 times out of 11 runs at a time. perhaps more training
