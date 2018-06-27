# CS 294-112 homework (offered in Fall of 2017)

This is my github repo for homework for [CS294](http://rail.eecs.berkeley.edu/deeprlcourse-fa17/index.html) (offered in Fall 2017). I covered this course remotely (using lecture notes and videos) and implemented the coding parts of the homework. Below are synopses for what I implemented for each homework assignment.  

*Disclaimer: this code is for **educational purposes** only.  Students taking current iterations of this course should refrain from copying this code, as that would breach academic integrity and hamper their own education.*

## Dependencies

* [Tensorflow 1.4](https://www.tensorflow.org/)
* [Numpy 1.13.3](http://www.numpy.org/)
* [Gym 0.10.5](https://gym.openai.com/) Gym 0.9.5 was used for homework 3.
* [Mujoco 1.5](http://www.mujoco.org/)

Note that some of these dependencies were not released at the time of this course. Furthermore, the starter code has been modified to reflect changes in OpenAI Gym's documentation.

## Homework 1

The course, up to this point, has covered more basic supervised learning. I implemented BC (behavior cloning) and DAgger (Dataset Aggregation), which improved the results (slightly). I also experimented with various hyperparameters. 

## Homework 2

I implemented the policy gradient algorithm and ran some tests on various environments. I played with the hyperparameters and saw that my implementation caused the agent's reward to converge to the theoretical value. I also implemented GAE (generalized advantage estimation) and compared its results. 

## Homework 3

I implemented the DQN algorithm and ran it on the Atari Pong simulator. I experimented with different hyperparameters and saw that my model converged to the perfect value.

## Homework 4

I implemented the MPC algorithm. However, I was unable to run the provided HalfCheetahEnvNew as it threw 

~~~~
'mujoco_py.cymj.PyMjModel' object has no attribute 'data'
~~~~

Furthermore, when I attempted to work with the given 'HalfCheetah-v2' environment that (in terms of raw code) is isomorphic to the HalfCheetahEnvNew, the action dimensions representing

~~~~
- rootx     slider      position (m)
- rootz     slider      position (m)
- rooty     hinge       angle (rad)
- bthigh    hinge       angle (rad)
- bshin     hinge       angle (rad)
- bfoot     hinge       angle (rad)
- fthigh    hinge       angle (rad)
- fshin     hinge       angle (rad)
- ffoot     hinge       angle (rad)
- rootx     slider      velocity (m/s)
- rootz     slider      velocity (m/s)
- rooty     hinge       angular velocity (rad/s)
- bthigh    hinge       angular velocity (rad/s)
- bshin     hinge       angular velocity (rad/s)
- bfoot     hinge       angular velocity (rad/s)
- fthigh    hinge       angular velocity (rad/s)
- fshin     hinge       angular velocity (rad/s)
- ffoot     hinge       angular velocity (rad/s)
~~~~

Aren't correctly represented in the loss function (the comments about what each part represents don't match up). Furthermore, for some strange reason, all HalfCheetah environments load in 17 variables, not 18. 