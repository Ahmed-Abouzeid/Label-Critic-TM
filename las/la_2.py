import numpy as np
import random

class LA_2:
    def __init__(self, n):
        self.n = n
        self.state = random.choice([self.n, self.n + 1]) 
    def reward(self):        
        if self.state <= self.n and self.state > 1: 
            self.state -= 1        
        elif self.state > self.n and self.state < 2 * self.n: 
            self.state += 1    
    def penalize(self):        
        if self.state <= self.n:        
            self.state += 1        
        elif self.state > self.n:            
            self.state -= 1    
    def get_label(self):        
        if self.state <= self.n:   
            return 0
        else:
            return 1