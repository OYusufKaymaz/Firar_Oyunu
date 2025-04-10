import numpy as np
from firar_env import Firar  

mahkum_q_table = np.load("q_table_mahkum.npy")  

env = Firar(render_mode="human")
state, _ = env.reset()

done = False
while not done:
    
    mahkum_action = np.argmax(mahkum_q_table[state])

    print("\nHarita:\n")
    print(env.render())  
    print("Gardiyan için bir hareket seçin (0: Güney, 1: Kuzey, 2: Doğu, 3: Batı):")
    gardiyan_action = int(input("Seçiminiz: "))  

    next_state, mahkum_reward, gardiyan_reward, done, _ = env.step((mahkum_action, gardiyan_action))

    print(f"\nMahkumun hareketi: {['Güney', 'Kuzey', 'Doğu', 'Batı'][mahkum_action]}")
    print(f"Gardiyanın hareketi: {['Güney', 'Kuzey', 'Doğu', 'Batı'][gardiyan_action]}")
    print(f"Mahkum ödülü: {mahkum_reward}, Gardiyan ödülü: {gardiyan_reward}")

    state = next_state

if mahkum_reward == 20:
    print("\nMahkum kaçtı! Gardiyan başarısız oldu.")
elif gardiyan_reward == 10:
    print("\nGardiyan mahkumu yakaladı! Tebrikler.")
