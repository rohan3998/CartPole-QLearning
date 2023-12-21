import gym
import numpy as np
import matplotlib.pyplot as plt
import time
#import pygame
import random
from collections import deque
from keras.models import *
from keras.layers import *
from keras.optimizers import Adam

from random import shuffle

env=gym.make('CartPole-v1')

max_iter = 500
alpha = 0.001
num_actions = env.action_space.n
num_observations = len(env.observation_space.high)
gamma = 0.96
memory_size = 10000
NUM_EPISODES = 180
minibatch = 64

epsilon_decay = 0.99
epsilon = 1
Samples=[]
Means=[]

def model_init():
  model = Sequential()
  model.add(Dense(24, activation='relu', input_shape=(num_observations, )))
  model.add(Dense(24, activation='relu'))
  model.add(Dense(2, activation='linear'))
  model.compile(
      optimizer=Adam(lr=alpha),
      loss = 'mse'
  )
  return model

def model_inference(model, obs):
  #print(obs)
  if type(obs) is tuple:
    obs = obs[0]
  np_obs = np.reshape(obs, [-1, num_observations])
  return model.predict(np_obs, verbose=0)

def train(model, x, y):
  obs = []
  for arr in x:
    if type(arr) is tuple:
      obs.append(arr[0])
    else:
      obs.append(arr)
  #print(obs)
  #print(type(x[0]))
  #print(type(y[0]))
  obs = np.reshape(obs, [-1, num_observations])
  target = np.reshape(y, [-1, num_actions])
  #print(x)
  #print(y)
  model.fit(obs, target, epochs=1, verbose=0)

def add_queue(deq, s, a, r, s_):
  if len(deq) > memory_size:
    if np.random.random()<0.5:
      shuffle(deq)
    deq.popleft()
  deq.append((s, a, r, s_))

def update_action(model, obs_batch):
  shuffle(obs_batch)
  batch_observations = []
  batch_targets = []

  for sample in obs_batch:
    s, a, r, s_ = sample
    q_est = model_inference(model, s)
    q_est = np.reshape(q_est, num_actions)
    q_est[a] = r
    if s_ is not None:
      pred = model_inference(model, s_)
      a_ = np.argmax(pred)
      q_est[a] += gamma * pred[0, a_]
    batch_observations.append(s)
    batch_targets.append(q_est)
  train(model, batch_observations, batch_targets)

deq = deque()

model = model_init()
print(model.summary())
epsilon = 1
for i in range(NUM_EPISODES):
  next_state = env.reset()
  done = False
  reward = 0
  action = np.random.randint(0,num_actions)
  epsilon = epsilon * epsilon_decay
  while(not done and reward < 500):
    current_state = next_state
    if np.random.random() > epsilon:
      action = np.argmax(model_inference(model,current_state))
    else:
      action = np.random.randint(0,num_actions)
    next_state, r, d, _, _ = env.step(action)
    done = d
    reward += r
    add_queue(deq, current_state, action, r, next_state)

    if len(deq) >= minibatch and np.random.random()<0.3 and Samples[-1]<500:
      sample_history = random.sample(deq,minibatch)
      update_action(model, sample_history)

  Samples.append(reward)

  print(('Episode:{}, iterations:{}').format(i,reward))

  if reward < 500:
    reward = -100
  add_queue(deq, current_state, action, reward, None)

print("Training Done")
plt.plot(Samples)
plt.title('Training Phase')
plt.ylabel('Time Steps')
plt.ylim(ymax=510)
plt.xlabel('Trial')
plt.savefig('Training.png', bbox_inches='tight')
plt.show()
