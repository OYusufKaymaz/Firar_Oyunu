import numpy as np
import random
from firar_env import Firar  


alpha = 0.1         
gamma = 0.3         
epsilon = 0.9       
epsilon_min = 0.01
epsilon_decay = 0.999999
episodes = 1000000



env = Firar(render_mode="ansi")


num_states = env.observation_space.n
num_actions = env.action_space.n  
num_joint_actions = num_actions * num_actions  


q_table = np.zeros((num_states, num_joint_actions))


mahkum_wins = 0
gardiyan_wins = 0

for episode in range(episodes):
    state, _ = env.reset()
    done = False
    episode_reward = 0  
    episode_mahkum_reward = 0
    episode_gardiyan_reward = 0

    while not done:
        
        if random.uniform(0, 1) < epsilon:
            joint_action = np.random.choice(num_joint_actions)
        else:
            joint_action = np.argmax(q_table[state])
        
        
        mahkum_action = joint_action // num_actions   
        gardiyan_action = joint_action % num_actions    
        
        
        next_state, mahkum_reward, gardiyan_reward, done, _ = env.step((mahkum_action, gardiyan_action))
        
        
        combined_reward = mahkum_reward - gardiyan_reward
        
        episode_reward += combined_reward
        episode_mahkum_reward += mahkum_reward
        episode_gardiyan_reward += gardiyan_reward
        
        
        best_next_joint_action = np.argmax(q_table[next_state])
        td_target = combined_reward + gamma * q_table[next_state, best_next_joint_action]
        td_error = td_target - q_table[state, joint_action]
        q_table[state, joint_action] += alpha * td_error
        
        state = next_state

    
    
    if mahkum_reward == 20:
        mahkum_wins += 1
    
    elif gardiyan_reward == 10:
        gardiyan_wins += 1

    if episode % 1000 == 0:
        print(f"Episode {episode}: Episode Combined Reward = {episode_reward}, "
              f"Mahkum Reward = {episode_mahkum_reward}, Gardiyan Reward = {episode_gardiyan_reward}, epsilon = {epsilon}")
    
    
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

print("\nEğitim tamamlandı!")
print(f"Mahkum kazandı: {mahkum_wins} kez")
print(f"Gardiyan kazandı: {gardiyan_wins} kez")


np.save("q_table_joint.npy", q_table)
print("Eğitim tamamlandı ve birleşik Q-table kaydedildi.")
