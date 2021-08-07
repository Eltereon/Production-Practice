# -*- coding: utf-8 -*-
"""
Created on Tue Jul 27 22:28:40 2021

@author: Eltereon
"""
from collections import Counter
import numpy as np
import random as random
import matplotlib.pyplot as plt

environment_states = 2 
q_values = np.zeros((environment_states, 2)) 

probabilities = [0.3,0.7, 
                 0.7,0.3, 
                 0.5,0.5, 
                 0.5,0.5]

rewards = np.array([[-2,3],
                    [-2,1],
                    [1,-2],
                    [0,-2]])

actions = ['strategy_1', 'strategy_2'] 
    
def get_starting_state():
    starting_state_index = np.random.randint(environment_states)
    return starting_state_index

def get_next_action(current_state_index, epsilon):  # фактор случайности выбора действия
    if np.random.random() < epsilon:
        return np.argmax(q_values[current_state_index])
    else: #выбирается случайное действие
        return np.random.randint(2)

def get_next_state(current_state_index, action_index):
  new_state_index = 0 
  reward = 0
  if current_state_index == 0 and actions[action_index] == 'strategy_1':
    new_state_index = random.choices([0,1], weights=[probabilities[0],probabilities[1]])
    reward = rewards[0,new_state_index]
  
  elif current_state_index == 0 and actions[action_index] == 'strategy_2':
    new_state_index = random.choices([0,1], weights=[probabilities[2],probabilities[3]])
    reward = rewards[1,new_state_index]

  if current_state_index == 1 and actions[action_index] == 'strategy_1':
    new_state_index = random.choices([0,1], weights=[probabilities[4],probabilities[5]])
    reward = rewards[2,new_state_index]
  
  elif current_state_index == 1 and actions[action_index] == 'strategy_2':
    new_state_index = random.choices([0,1], weights=[probabilities[6],probabilities[7]])
    reward = rewards[3,new_state_index]
  
  return new_state_index, reward
    
# print (Counter(random.choices([0, 1], weights=[0.3, 0.7])[0]
#         for _ in range(100000)))   # проверка работоспособности перехода через вероятности (веса)

total_rewards0 = np.zeros((1000))
iterator0 = 0
for episode in range(999):#тренировка агента
  epsilon = 0.9
  discount_factor = 0.5
  learning_rate = 0.3
  state_index = 0
  cycle_step = 0
  sum_rewards = 0
  iterator0 += 1
  while cycle_step < 30: 
    action_index = get_next_action(state_index, epsilon) 
    old_state_index = state_index 
    state_index, reward = get_next_state(state_index, action_index) 
    old_q_value = q_values[old_state_index, action_index]
    temporal_difference = reward + (discount_factor * np.max(q_values[state_index])) - old_q_value   
    new_q_value = old_q_value + (learning_rate * temporal_difference)
    q_values[old_state_index, action_index] = new_q_value
    sum_rewards +=reward
    total_rewards0[iterator0]=sum_rewards
    cycle_step += 1

 
for i in range(2):
    print()

print("Q-matrix:")

for i in range(2):
    for j in range(2):
        print(q_values[i][j], end = ' ')
    print()

for i in range(2):
    print()
    
print("Best solutions:")
print()
print('In state 1:','', np.argmax(q_values[0])+1)
print('In state 2:','', np.argmax(q_values[1])+1)

def final_score():#Проход одного эпизода обученным агентом
    epsilon = 1
    state_index = 1
    cycle_steps = 0
    sum_rewards = 0
    while cycle_steps < 30: 
      action_index = get_next_action(state_index, epsilon)
      state_index, reward = get_next_state(state_index, action_index) 
      cycle_steps += 1
      sum_rewards+=reward
    return sum_rewards

total_rewards1 = np.zeros((1000)) #проход множества эпизодов без обучения (рандомно)
iterator1 = 0
for episode in range(999):
  epsilon = 0
  state_index = 0
  cycle_step = 0
  sum_rewards = 0
  iterator1 += 1
  while cycle_step < 30: 
    action_index = get_next_action(state_index, epsilon) 
    old_state_index = state_index 
    state_index, reward = get_next_state(state_index, action_index)    
    sum_rewards +=reward
    total_rewards1[iterator1]=sum_rewards
    cycle_step += 1


total_rewards2 = np.zeros((1000)) #проход множества эпизодов обученным агентом с сохранением сумм ревардов (чтобы потом увидеть дифференс на графиках)
iterator2 = 0
for episode in range(999):
  epsilon = 1
  state_index = 0
  cycle_step = 0
  sum_rewards = 0
  iterator2 += 1
  while cycle_step < 30: 
    action_index = get_next_action(state_index, epsilon) 
    old_state_index = state_index 
    state_index, reward = get_next_state(state_index, action_index)    
    sum_rewards +=reward
    total_rewards2[iterator2]=sum_rewards
    cycle_step += 1


plt.plot(total_rewards0)
plt.show()
plt.plot(total_rewards1)
plt.show()
plt.plot(total_rewards2)
plt.show()





    



