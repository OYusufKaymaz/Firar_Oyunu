from typing import Optional
import numpy as np
from gym import Env, spaces
import pygame 

class Firar(Env):
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None):
        MAP = [
            "+=+=+=+=+=+=+=+=+",
            "| : | : | : | : |",
            "+=+-+-+=+-+-+=+-+",
            "| : | | : | : : |",
            "+-+=+-+-+=+=+=+-+",
            "| : | : | : : : |",
            "+-+-+=+-+-+-+-+=+",
            "| | : : : | | : |",
            "+-+=+=+=+-+-+=+-+",
            "| |H: : : | : | |",
            "+-+=+-+-+-+=+=+-+",
            "| : : | | : | : |",
            "+=+=+=+-+=+-+-+-+",
            "| : | : | | : | |",
            "+-+-+-+-+-+-+-+-+",
            "| | : : : : : : |",
            "+=+=+=+=+=+=+=+=+",
        ]

        LOCS = [(0, 3), (0, 6), (4, 1)]

        
        num_rows = 8  
        num_columns = 8  
        num_states = (num_rows * num_columns) * (num_rows * num_columns)
        
        
        pygame.init()
        self.WIDTH, self.HEIGHT = 600, 600
        self.GRID_SIZE = 8
        self.CELL_SIZE = self.WIDTH // self.GRID_SIZE
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Firar Oyunu")

        
        # self.AGENT_COLOR = (0, 255, 0)  
        # self.GUARD_COLOR = (255, 0, 0)  
        # self.EXIT_COLOR = (0, 0, 255)  
        self.BACKGROUND_COLOR = (30, 30, 30)
        
        
        self.desc = np.asarray(MAP, dtype="c")  
        self.locs = LOCS  

        num_actions = 4  

        self.initial_state_distrib = np.zeros(num_states)
        self.observation_space = spaces.Discrete(num_states)
        self.action_space = spaces.Discrete(num_actions)

        
        self.render_mode = render_mode
        self.s = None
        self.lastaction = None

        
        self.P = {}
        for mahkum_row in range(num_rows):
            for mahkum_col in range(num_columns):
                for gardiyan_row in range(num_rows):
                    for gardiyan_col in range(num_columns):
                        state = self.encode((mahkum_row, mahkum_col), (gardiyan_row, gardiyan_col))
                        self.P[state] = {(mahkum_action, gardiyan_action): [] for mahkum_action in range(4) for gardiyan_action in range(4)}
        
                        for mahkum_action in range(4):
                            for gardiyan_action in range(4):
                                new_mahkum_row, new_mahkum_col = mahkum_row, mahkum_col
                                new_gardiyan_row, new_gardiyan_col = gardiyan_row, gardiyan_col
        
                                
                                if mahkum_action == 0:  
                                    if self.desc[2 * mahkum_row + 2, 2 * mahkum_col + 1] == b"-":  
                                        new_mahkum_row = mahkum_row + 1
                                elif mahkum_action == 1:  
                                    if self.desc[2 * mahkum_row, 2 * mahkum_col + 1] == b"-":  
                                        new_mahkum_row = mahkum_row - 1
                                elif mahkum_action == 2:  
                                    if self.desc[2 * mahkum_row + 1, 2 * mahkum_col + 2] == b":":  
                                        new_mahkum_col = mahkum_col + 1
                                elif mahkum_action == 3:  
                                    if self.desc[2 * mahkum_row + 1, 2 * mahkum_col] == b":":  
                                        new_mahkum_col = mahkum_col - 1

                                
                                if gardiyan_action == 0:  
                                    if self.desc[2 * gardiyan_row + 2, 2 * gardiyan_col + 1] == b"-":  
                                        new_gardiyan_row = gardiyan_row + 1
                                elif gardiyan_action == 1:  
                                    if self.desc[2 * gardiyan_row, 2 * gardiyan_col + 1] == b"-":  
                                        new_gardiyan_row = gardiyan_row - 1
                                elif gardiyan_action == 2:  
                                    if self.desc[2 * gardiyan_row + 1, 2 * gardiyan_col + 2] == b":":  
                                        new_gardiyan_col = gardiyan_col + 1
                                elif gardiyan_action == 3:  
                                    if self.desc[2 * gardiyan_row + 1, 2 * gardiyan_col] == b":":  
                                        new_gardiyan_col = gardiyan_col - 1

        
                                
                                mahkum_reward = -1
                                gardiyan_reward = -1
                                terminated = False
        
                                if (new_mahkum_row, new_mahkum_col) == (new_gardiyan_row, new_gardiyan_col):
                                    mahkum_reward = -10  
                                    gardiyan_reward = 10
                                    terminated = True
                                elif (new_mahkum_row, new_mahkum_col) == self.locs[2]:
                                    mahkum_reward = 20  
                                    gardiyan_reward = -10
                                    terminated = True
        
                                
                                new_state = self.encode((new_mahkum_row, new_mahkum_col), (new_gardiyan_row, new_gardiyan_col))
                                self.P[state][(mahkum_action, gardiyan_action)].append((1.0, new_state, mahkum_reward, gardiyan_reward, terminated))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        """Ortama başlangıç durumunu atar."""
        super().reset(seed=seed)
        mahkum_start = self.locs[0]  
        gardiyan_start = self.locs[1]  
    
        self.mahkum_pos = mahkum_start
        self.gardiyan_pos = gardiyan_start
    
        self.s = self.encode(self.mahkum_pos, self.gardiyan_pos)
        assert self.s in self.P, f"Başlangıç durumu {self.s} geçiş tablosunda yok."
        self.lastaction = None
    
        return self.s, {}

    def step(self, action):
        """
        Mahkum ve gardiyanın hareketlerini ve oyun durumunu günceller.
        """
        mahkum_action, gardiyan_action = action
        transitions = self.P[self.s][(mahkum_action, gardiyan_action)]
        
        
        
        i = self.np_random.choice(len(transitions))
        prob, next_state, mahkum_reward, gardiyan_reward, done = transitions[i]
    
        
        self.s = next_state
    
        
        self.mahkum_pos, self.gardiyan_pos = self.decode(self.s)
    
        return self.s, mahkum_reward, gardiyan_reward, done, {}
    
    
    def guard_step(self, gardiyan_action):
        """
        Sadece gardiyanın hareketini uygular.
        Mahkum pozisyonu değişmez.
        Dönüş: (new_state, dummy_prisoner_reward, gardiyan_reward, done, extra_info)
        """
        
        row, col = self.gardiyan_pos
        new_row, new_col = row, col
    
        if gardiyan_action == 0:  
            if self.desc[2 * row + 2, 2 * col + 1] == b"-":
                new_row = row + 1
        elif gardiyan_action == 1:  
            if self.desc[2 * row, 2 * col + 1] == b"-":
                new_row = row - 1
        elif gardiyan_action == 2:  
            if self.desc[2 * row + 1, 2 * col + 2] == b":":
                new_col = col + 1
        elif gardiyan_action == 3:  
            if self.desc[2 * row + 1, 2 * col] == b":":
                new_col = col - 1
    
        self.gardiyan_pos = (new_row, new_col)
        
        self.s = self.encode(self.mahkum_pos, self.gardiyan_pos)
    
        
        dummy_prisoner_reward = -1  
        gardiyan_reward = -1
        done = False
    
        
        if self.gardiyan_pos == self.mahkum_pos:
            done = True
            gardiyan_reward = 10
            dummy_prisoner_reward = -10
    
        return self.s, dummy_prisoner_reward, gardiyan_reward, done, {}
    
    
    def prisoner_step(self, mahkum_action):
        """
        Sadece mahkumun hareketini uygular.
        Gardiyan pozisyonu değişmez.
        
        Dönüş: (new_state, mahkum_reward, dummy_guard_reward, done, extra_info)
        """
        row, col = self.mahkum_pos
        new_row, new_col = row, col
        
        
        if mahkum_action == 0:  
            if self.desc[2 * row + 2, 2 * col + 1] == b"-":
                new_row = row + 1
        elif mahkum_action == 1:  
            if self.desc[2 * row, 2 * col + 1] == b"-":
                new_row = row - 1
        elif mahkum_action == 2:  
            if self.desc[2 * row + 1, 2 * col + 2] == b":":
                new_col = col + 1
        elif mahkum_action == 3:  
            if self.desc[2 * row + 1, 2 * col] == b":":
                new_col = col - 1
        
        
        self.mahkum_pos = (new_row, new_col)
        
        self.s = self.encode(self.mahkum_pos, self.gardiyan_pos)
        
        dummy_guard_reward = -1
        mahkum_reward = -1
        done = False
        
        
        if self.mahkum_pos == self.locs[2]:
            done = True
            mahkum_reward = 20
            dummy_guard_reward = -10
        
        
        
        elif self.mahkum_pos == self.gardiyan_pos:
            done = True
            mahkum_reward = -10      
            dummy_guard_reward = 10  
        
        return self.s, mahkum_reward, dummy_guard_reward, done, {}



    
    def render(self):
        """Durumu render eder (sadece ANSI formatında)."""
        if self.render_mode == "ansi":
            
            desc = self.desc.tolist()
            desc = [[c.decode("utf-8") for c in line] for line in desc]
    
            
            for row in range(len(desc)):
                for col in range(len(desc[row])):
                    if desc[row][col] in ["M", "G"]:
                        desc[row][col] = " "  
    
            
            mahkum_row, mahkum_col = self.mahkum_pos
            gardiyan_row, gardiyan_col = self.gardiyan_pos
    
            
            print(f"Mahkum Pozisyonu (row, col): ({mahkum_row}, {mahkum_col})")
            print(f"Gardiyan Pozisyonu (row, col): ({gardiyan_row}, {gardiyan_col})")
    
            
            try:
                
                desc[1 + 2 * mahkum_row][2 * mahkum_col + 1] = "M"
                
                desc[1 + 2 * gardiyan_row][2 * gardiyan_col + 1] = "G"
            except IndexError:
                print("Pozisyonlar harita sınırları dışında!")
    
            
            rendered_map = "\n".join("".join(line) for line in desc) + "\n"
            return rendered_map
        
        """Oyun durumunu Pygame ile görselleştirir."""
        if self.render_mode == "human":
            
            self.screen.fill((0, 0, 0))  
            
            
            self.draw_map()
    
            
            self.draw_entities()
    
            
            pygame.display.flip()





    def encode(self, mahkum_pos, gardiyan_pos):
        """Durumu kodlar."""
        mahkum_row, mahkum_col = mahkum_pos
        gardiyan_row, gardiyan_col = gardiyan_pos
        assert 0 <= mahkum_row < 8 and 0 <= mahkum_col < 8, "Mahkum pozisyonu geçersiz!"
        assert 0 <= gardiyan_row < 8 and 0 <= gardiyan_col < 8, "Gardiyan pozisyonu geçersiz!"
        return ((mahkum_row * 8 + mahkum_col) * 64 + gardiyan_row * 8 + gardiyan_col)

    def decode(self, s):
        """Kodlanmış durumu çöz."""
        gardiyan_col = s % 8  
        s //= 8
        gardiyan_row = s % 8  
        s //= 8
        mahkum_col = s % 8  
        s //= 8
        mahkum_row = s  
        return (mahkum_row, mahkum_col), (gardiyan_row, gardiyan_col)

    def draw_map(self):
        """
        ASCII haritayı (self.desc) kullanarak Pygame'de duvarları çizer.
        - 8x8 hücrenin her biri için üst/alt/sol/sağ duvar var mı kontrol eder.
        - Duvar varsa çizgi çekerek ayırt edilmesini sağlar.
        """
        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                
                x = col * self.CELL_SIZE
                y = row * self.CELL_SIZE
    
                
                pygame.draw.rect(
                    self.screen, 
                    self.BACKGROUND_COLOR, 
                    (x, y, self.CELL_SIZE, self.CELL_SIZE)
                )
    
                
                
                if not (self.desc[2 * row, 2 * col + 1] == b"-"):
                    pygame.draw.line(self.screen, (255, 255, 255), (x, y), (x + self.CELL_SIZE, y), 2)
                
                
                if not (self.desc[2 * row + 2, 2 * col + 1] == b"-"):
                    pygame.draw.line(self.screen, (255, 255, 255), (x, y + self.CELL_SIZE), (x + self.CELL_SIZE, y + self.CELL_SIZE), 2)
                
                
                if not (self.desc[2 * row + 1, 2 * col] == b":"):
                    pygame.draw.line(self.screen, (255, 255, 255), (x, y), (x, y + self.CELL_SIZE), 2)
                
                
                if not (self.desc[2 * row + 1, 2 * col + 2] == b":"):
                    pygame.draw.line(self.screen, (255, 255, 255), (x + self.CELL_SIZE, y), (x + self.CELL_SIZE, y + self.CELL_SIZE), 2)

    
    def draw_entities(self):
        
        mx = self.mahkum_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        my = self.mahkum_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
    
        
        gx = self.gardiyan_pos[1] * self.CELL_SIZE + self.CELL_SIZE // 2
        gy = self.gardiyan_pos[0] * self.CELL_SIZE + self.CELL_SIZE // 2
    
        
        pygame.draw.circle(self.screen, (255, 0, 0), (mx, my), self.CELL_SIZE // 3)
    
        
        pygame.draw.rect(
            self.screen, (0, 0, 255), (gx - self.CELL_SIZE // 4, gy - self.CELL_SIZE // 4, self.CELL_SIZE // 2, self.CELL_SIZE // 2)
        )
