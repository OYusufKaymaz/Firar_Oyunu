import numpy as np
import pygame
from firar_env import Firar  


mahkum_q_table = np.load("q_table_mahkum.npy")  


env = Firar(render_mode="human")
state, _ = env.reset()

done = False
clock = pygame.time.Clock()

while not done:
    guard_action = None

    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                guard_action = 3  # Batı
            elif event.key == pygame.K_RIGHT:
                guard_action = 2  # Doğu
            elif event.key == pygame.K_UP:
                guard_action = 1  # Kuzey
            elif event.key == pygame.K_DOWN:
                guard_action = 0  # Güney

    
    if guard_action is None:
        env.render()
        clock.tick(10)
        continue

    
    state, _, gardiyan_reward, done, _ = env.guard_step(guard_action)
    
    env.render()
    
    if done:
        break
    
    pygame.time.delay(100)  
    
    
    mahkum_action = np.argmax(mahkum_q_table[state])
    state, mahkum_reward, _, done, _ = env.prisoner_step(mahkum_action)
    pygame.time.delay(100)  

    env.render()
    clock.tick(10)

env.close()
pygame.quit()


if mahkum_reward == 20:
    print("\nMahkum kaçtı! Gardiyan başarısız oldu.")
elif gardiyan_reward == 10:
    print("\nGardiyan mahkumu yakaladı! Tebrikler.")
