# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 22:28:40 2021

@author: Eltereon
"""

import numpy as np
import random as random

environment_states = 2 # Состояния
q_values = np.zeros((environment_states, 2)) # матрица значений Q изначально нулевая

probabilities = [0.3,0.7, 0.7,0.3, 0.5,0.5, 0.5,0.5]
rewards = [-10,15,-10,5,5,-10,2,-10]

actions = ['strategy_1', 'strategy_2'] 
    
def get_starting_state():
    starting_state_index = np.random.randint(environment_states)
    return starting_state_index

def get_next_action(current_state_index, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_state_index])
    else: #выбирается случайное действие
        return np.random.randint(2)


def get_next_state(current_state_index, action_index):
  temp_probability = np.random.random()
  new_state_index = 0 
  reward = 0
  if current_state_index == 0 and actions[action_index] == 'strategy_1' and temp_probability < probabilities[0]:
    new_state_index = 0
    reward = rewards[0]
  elif current_state_index == 0 and actions[action_index] == 'strategy_1' and temp_probability > probabilities[0]:
    new_state_index = 1
    reward = rewards[1]
  elif current_state_index == 0 and actions[action_index] == 'strategy_2' and temp_probability < probabilities[2]:
    new_state_index = 0
    reward = rewards[2]
  elif current_state_index == 0 and actions[action_index] == 'strategy_2' and temp_probability > probabilities[2]:
    new_state_index = 1
    reward = rewards[3]
  if current_state_index == 1 and actions[action_index] == 'strategy_1' and temp_probability < probabilities[4]:
    new_state_index = 0
    reward = rewards[4]
  elif current_state_index == 1 and actions[action_index] == 'strategy_1' and temp_probability > probabilities[4]:
    new_state_index = 1
    reward = rewards[5]
  elif current_state_index == 1 and actions[action_index] == 'strategy_2' and temp_probability < probabilities[6]:
    new_state_index = 0
    reward = rewards[6]
  elif current_state_index == 1 and actions[action_index] == 'strategy_2' and temp_probability > probabilities[6]:
    new_state_index = 1 
    reward = rewards[7]
  return new_state_index, reward

epsilon = 0.8
discount_factor = 0.5
learning_rate = 0.1
cycle_step = 0


for episode in range(100000):
  state_index = get_starting_state()
  while cycle_step < 1000: 
    action_index = get_next_action(state_index, epsilon) 
    old_state_index = state_index 
    state_index, reward = get_next_state(state_index, action_index) 
    old_q_value = q_values[old_state_index, action_index]
    temporal_difference = reward + (discount_factor * np.max(q_values[state_index])) - old_q_value   
    new_q_value = old_q_value + (learning_rate * temporal_difference)
    q_values[old_state_index, action_index] = new_q_value
    cycle_step += 1

for i in range(4):
    print()

print("Q-matrix:")

for i in range(2):
    for j in range(2):
        print(q_values[i][j], end = ' ')
    print()

for i in range(2):
    print()
    
print("Best solutions")
print()
print('In state 1:','', np.argmax(q_values[0])+1)
print('In state 2:','', np.argmax(q_values[1])+1 )

    
# Заметка: для рандома можно использовать веса

