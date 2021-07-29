# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 22:28:40 2021

@author: Eltereon
"""

import numpy as np
import random as random

environment_states = 2 # Состояния
q_values = np.zeros((environment_states, 2)) # матрица значений Q

probabilities = [0.7,0.3, 0.5,0.5, 0.6,0.4, 0.8,0.2]
rewards = [10,6, 5,4, 4,-2, 1,-5]

actions = ['strategy_1', 'strategy_2']

def is_finish(cycle_step):
    if cycle_step == 1000.:
        return True
    else:
        return False
    
def get_starting_state():
    current_state_index = np.random.randint(environment_states)

def get_next_action(current_state_index, epsilon):
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_state_index])
    else: #выбирается случайное действие
        return np.random.randint(2)

def get_next_state(current_state_index, action_index):
  new_state_index = current_state_index
  temp_probability = np.random.random()
  if current_state_index == 1 and actions[action_index] == 'strategy_1' and np.random.random() > probabilities[0]:
    new_state_index == 1

  elif current_state_index == 1 and actions[action_index] == 'strategy_2':
    new_row_index += 1

  return new_state_index

# Заметка ? реализовать случ.переход по вероятностям через веса random.choises или как задумал ?

