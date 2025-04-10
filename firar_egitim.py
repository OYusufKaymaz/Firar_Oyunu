import numpy as np
import random
from firar_env import Firar  

# Q-learning Hyperparameters
alpha = 0.1  # Öğrenme katsayısı
gamma = 0.3  # İndirim faktörü
epsilon = 0.9  
epsilon_min = 0.01
epsilon_decay = 0.999
episodes = 1000000

# Ortamı başlat
env = Firar(render_mode="ansi")

# Q-table'ları başlat
num_states = env.observation_space.n
num_actions = env.action_space.n
q_table_mahkum = np.zeros((num_states, num_actions))  # Mahkumun Q-table'ı
q_table_gardiyan = np.zeros((num_states, num_actions))  # Gardiyanın Q-table'ı

# Kazanma sayacı
mahkum_wins = 0
gardiyan_wins = 0

for episode in range(episodes):
    
    state, _ = env.reset()
    done = False
    episode_mahkum_reward = 0
    episode_gardiyan_reward = 0

    while not done:
        # Mahkum ve gardiyan için epsilon-greedy aksiyon seçimi
        if random.uniform(0, 1) < epsilon:
            mahkum_action = np.random.choice(num_actions)  
        else:
            mahkum_action = np.argmax(q_table_mahkum[state])  

        if random.uniform(0, 1) < epsilon:
            gardiyan_action = np.random.choice(num_actions)  
        else:
            gardiyan_action = np.argmax(q_table_gardiyan[state])  

        # Ortamda bir adım at
        next_state, mahkum_reward, gardiyan_reward, done, _ = env.step((mahkum_action, gardiyan_action))

        # Ödülleri güncelle
        episode_mahkum_reward += mahkum_reward
        episode_gardiyan_reward += gardiyan_reward

        # Q-table güncellemeleri
        best_next_action_mahkum = np.argmax(q_table_mahkum[next_state])
        q_table_mahkum[state, mahkum_action] = q_table_mahkum[state, mahkum_action] + alpha * (
            mahkum_reward + gamma * q_table_mahkum[next_state, best_next_action_mahkum] - q_table_mahkum[state, mahkum_action]
        )

        best_next_action_gardiyan = np.argmax(q_table_gardiyan[next_state])
        q_table_gardiyan[state, gardiyan_action] = q_table_gardiyan[state, gardiyan_action] + alpha * (
            gardiyan_reward + gamma * q_table_gardiyan[next_state, best_next_action_gardiyan] - q_table_gardiyan[state, gardiyan_action]
        )

        # Durumu güncelle
        state = next_state
        
        if episode == 2000:
            print(env.render())

    # Kazananı belirle
    if mahkum_reward == 20:  
        mahkum_wins += 1
    elif gardiyan_reward == 10:  
        gardiyan_wins += 1

    
    if episode % 1000 == 0:
        print(f"Episode {episode}: Mahkum Reward = {episode_mahkum_reward}, Gardiyan Reward = {episode_gardiyan_reward}, epsilon= {epsilon}")
    
    # Epsilon değerini azalt
    if episode % 1000 == 0 and epsilon > epsilon_min:
        epsilon *= epsilon_decay


print("\nEğitim tamamlandı!")
print(f"Mahkum kazandı: {mahkum_wins} kez")
print(f"Gardiyan kazandı: {gardiyan_wins} kez")

# Eğitim tamamlandıktan sonra Q-table'ları kaydetme
np.save("q_table_mahkum.npy", q_table_mahkum)
np.save("q_table_gardiyan.npy", q_table_gardiyan)

print("Eğitim tamamlandı ve Q-table'lar kaydedildi.")
