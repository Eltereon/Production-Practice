!pip install -q tf-agents

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt

import abc
import tensorflow as tf
import numpy as np

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.reset_default_graph
inputs1 = tf.compat.v1.placeholder(shape=[1,2],dtype=tf.float32)
W = tf.Variable(tf.compat.v1.random_uniform([2,2]))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

nextQ = tf.compat.v1.placeholder(shape=[1,2],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.compat.v1.initialize_all_variables()

class MyEnvironment(py_environment.PyEnvironment):

  def __init__(self):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(1,), dtype=np.int32, minimum=0, name='observation')
    self._state = 0
    self._episode_ended = False

  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = 0
    self._episode_ended = False
    return ts.restart(np.array([self._state], dtype=np.int32))
  
  def _step(self, action):
    reward = 0   
    testArray = np.array([
                      [[0.3,0.7], [0.7,0.3], [[-10,15], [-10,5]]],
                      [[0.5,0.5], [0.5,0.5], [[5,-10], [2,-10]]]
                    ]) 
    weights = testArray[self._state][action]
    current_state = self._state
    length = len(testArray[self._state]) - 1
    self._state = random.choices([0,1], weights)[0]
    reward = testArray[current_state][length][action][self._state]

    return ts.termination(np.array([self._state], dtype=np.int32), reward)
         
environment = MyEnvironment()
#action1 = 0
#action2 = 1
#time_step = environment.reset()
#print(time_step)
#cumulative_reward = time_step.reward
#cycle_step = 0
#for _ in range(29):
#  time_step = environment.step(0)
#  print(time_step.observation)
#  cumulative_reward += time_step.reward
#print('Final Reward = ', cumulative_reward)

discount_factor = .99
num_episodes = 1000
total_rewards = []

def Go(epsilon):
    e = epsilon
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            time_step = environment.reset()
            reward_sum = 0
            j = 0
            while j < 30:
                j+=1
                s = time_step.observation[0]
                a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(2)[s:s+1]})
                if np.random.rand(1) > e:
                    a[0] = np.random.randint(2)
                time_step = environment.step(a[0])
                reward = time_step.reward
                s1 = time_step.observation[0]
                Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(2)[s1:s1+1]})
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,a[0]] = reward + discount_factor*maxQ1
                W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(2)[s:s+1],nextQ:targetQ})
                reward_sum += time_step.reward
                
            total_rewards.append(reward_sum)

Go(0.9)
plt.plot(total_rewards)
plt.show
print(np.mean(total_rewards))
Go(0)
plt.plot(total_rewards)
plt.show
print(np.mean(total_rewards))
