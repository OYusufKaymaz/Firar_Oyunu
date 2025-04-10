import numpy as np
import pygame
from firar_env import Firar


q_table = np.load("q_table_joint.npy")


env = Firar(render_mode="human")
state, _ = env.reset()
done = False
clock = pygame.time.Clock()

while not done:
    g_action = None

    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                g_action = 3  
            elif event.key == pygame.K_RIGHT:
                g_action = 2  
            elif event.key == pygame.K_UP:
                g_action = 1  
            elif event.key == pygame.K_DOWN:
                g_action = 0  

    
    if g_action is None:
        env.render()
        clock.tick(10)
        continue

    
    state, dummy_guard_reward, guard_reward, done, _ = env.guard_step(g_action)
    

    
    if done:
        break
    env.render()
    pygame.time.delay(100)  
    
    
    possible_q = []
    for p_action in range(4):
        joint_action = p_action * 4 + g_action
        possible_q.append(q_table[state, joint_action])
    p_action = int(np.argmax(possible_q))
    
    
    state, p_reward, dummy_p_reward, done, _ = env.prisoner_step(p_action)
    

    
    print(f"Gardiyan hareketi: {['Güney', 'Kuzey', 'Doğu', 'Batı'][g_action]}")
    print(f"Mahkum hareketi: {['Güney', 'Kuzey', 'Doğu', 'Batı'][p_action]}")
    print(f"Gardiyan ödülü: {guard_reward}, Mahkum ödülü: {p_reward}\n")

    
    env.render()
    clock.tick(10)
    pygame.time.delay(100)

env.close()
pygame.quit()


if p_reward == 20:
    print("\nMahkum kaçtı! Gardiyan başarısız oldu.")
elif guard_reward == 10:
    print("\nGardiyan mahkumu yakaladı! Tebrikler.")
