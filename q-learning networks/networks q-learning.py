# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 19:48:02 2021

@author: lancaster
"""
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt

environment_states = 2

testArray = np.array([
                      [[0.3,0.7], [0.7,0.3], [[-10,15], [-10,5]]],
                      [[0.5,0.5], [0.5,0.5], [[5,-10], [2,-10]]]
                    ])

tf.compat.v1.reset_default_graph
tf.compat.v1.disable_eager_execution()
inputs1 = tf.compat.v1.placeholder(shape=[1,2],dtype=tf.float32)
W = tf.Variable(tf.compat.v1.random_uniform([2,2]))
Qout = tf.matmul(inputs1,W)
predict = tf.argmax(Qout,1)

nextQ = tf.compat.v1.placeholder(shape=[1,2],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))
trainer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)
updateModel = trainer.minimize(loss)

init = tf.compat.v1.initialize_all_variables()

discount_factor = .99
num_episodes = 100

total_rewards = []

def get_next_state(current_state_index, action_index):
  weights = testArray[current_state_index][action_index]
  length = len(testArray[current_state_index]) - 1
  new_state_index = random.choices([0,1], weights)[0]
  reward = testArray[current_state_index][length][action_index][new_state_index]
  return new_state_index, reward

def Go(epsilon):
    e = epsilon
    with tf.compat.v1.Session() as sess:
        sess.run(init)
        for i in range(num_episodes):
            s = 0
            reward_sum = 0
            j = 0
            while j < 30:
                j+=1
                a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(2)[s:s+1]})
                if np.random.rand(1) > e:
                    a[0] = np.random.randint(2)
                s1,reward = get_next_state(s,a[0])               
                Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(2)[s1:s1+1]})
                maxQ1 = np.max(Q1)
                targetQ = allQ
                targetQ[0,a[0]] = reward + discount_factor*maxQ1
                W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(2)[s:s+1],nextQ:targetQ})
                reward_sum += reward
                s = s1
            total_rewards.append(reward_sum)

Go(0.9)
plt.plot(total_rewards)
plt.show
Go(0)
plt.plot(total_rewards)
plt.show
