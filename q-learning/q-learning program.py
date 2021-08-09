# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 22:28:40 2021

@author: Eltereon
"""
import numpy as np
import random as random
import matplotlib.pyplot as plt

def get_starting_state():
    starting_state_index = np.random.randint(environment_states)
    return starting_state_index

def get_next_action(current_state_index, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_state_index])
    else: 
        return np.random.randint(2)
    
def get_next_state(current_state_index, action_index):
  weights = testArray[current_state_index][action_index]
  length = len(testArray[current_state_index]) - 1
  new_state_index = random.choices([0,1], weights)[0]
  reward = testArray[current_state_index][length][action_index][new_state_index]
  return new_state_index, reward

def get_total_rewards(epsilon):
    total_rewards = np.zeros((1000)) # Прогон без обучения (eps = 0), прогон обученным агентом (eps = 1) - получениt сумм ревардов за каждый эпизод
    for i in range(999):
      state_index = cycle_step = sum_rewards = 0
      while cycle_step < days_count - 1: 
        action_index = get_next_action(state_index, epsilon)
        state_index, reward = get_next_state(state_index, action_index)    
        sum_rewards +=reward
        total_rewards[i]=sum_rewards
        cycle_step += 1
    return total_rewards

environment_states = 2
q_values = np.zeros((environment_states, 2)) 

testArray = np.array([
                      [[0.3,0.7], [0.7,0.3], [[-10,15], [-10,5]]],
                      [[0.5,0.5], [0.5,0.5], [[5,-10], [2,-10]]]
                    ])

epsilon = 0.4
discount_factor = 0.9
learning_rate = 0.3

days_count = 30

for episode in range(10000):
  state_index = cycle_step = sum_rewards = 0  
  while cycle_step < days_count - 1: 
    action_index = get_next_action(state_index, epsilon) 
    old_state_index = state_index 
    state_index, reward = get_next_state(state_index, action_index) 
    old_q_value = q_values[old_state_index, action_index]
    temporal_difference = reward + (discount_factor * np.max(q_values[state_index])) - old_q_value   
    new_q_value = old_q_value + (learning_rate * temporal_difference)
    q_values[old_state_index, action_index] = new_q_value
    cycle_step += 1

print('\n')
print("Q-matrix:")

for i in range(2):
    for j in range(2):
        print(q_values[i][j], end = ' ')
    print()
print('\n')
print("Best solutions:\n")
print('In state 1:','', np.argmax(q_values[0])+1)
print('In state 2:','', np.argmax(q_values[1])+1)

total_rewards1 = get_total_rewards(0)
total_rewards2 = get_total_rewards(1)

plt.plot(total_rewards1)
plt.show()
plt.plot(total_rewards2)
plt.show()

print(np.mean(total_rewards1))
print(np.mean(total_rewards2))
