#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates Lycacode layout figure

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

import pygame
import numpy as np

S = 600

class Lycacode:
    RES = 3
    
    def __init__(self, *args):
        self.create_matrix()
    
    def create_matrix(self):
        R = self.RES
        self.matrix = np.zeros([R*9, R*9])-3
        self.matrix[R*4:R*5, :] = -1
        self.matrix[:, R*4:R*5] = -1
        self.matrix[R:R*2, R*3:R*6] = -1
        self.matrix[R*3:R*6, R:R*2] = -1
        self.matrix[-R*2:-R, -R*6:-R*3] = -1
        self.matrix[-R*6:-R*3, -R*2:-R] = -1

        self.matrix[R*4:R*5,R*4:R*5] = 0
        self.matrix[0, R*4:R*5] = -2 # mask
        self.matrix[-1, R*4:R*5] = -2 # mask
        self.matrix[R*4, R*4+1:R*5-1] = 1
        self.matrix[R*5-1, R*4] = 1
        self.matrix[R*4+1:R*5-1,R*4+1:R*5-1] = 1
    
    def display(self, surf):
        R = self.RES
        S = min(surf.get_size())
        s = int(S/12/R)*R
        O = (S-s*9)/2

        surf.fill((255,255,255))
        
        for y in range(R*9):
            for x in range(R*9):
                col = self.matrix[y,x]
                if col == -3:
                    X, Y = O+x*s/R, O+y*s/R
                    size = s/R
                    pygame.draw.line(surf, (0,0,0), [X+size/2, Y], [X, Y+size/2])
                    pygame.draw.line(surf, (0,0,0), [X+size, Y+size/2], [X+size/2, Y+size])
                else:
                    if col == -2:
                        col = (190,190,190)
                    elif col == -1:
                        col = (127,127,127)
                    elif col == 0:
                        col = (0,0,0)
                    elif col == 1:
                        col = (255,255,255)
                    pygame.draw.rect(surf, col, [O+x*s/R, O+y*s/R, s/R, s/R])

if __name__ == "__main__":
    pygame.init()
    
    surf = pygame.Surface([S, S])
    
    code = Lycacode()
    code.display(surf)
    pygame.image.save(surf, "lycacode_layout.png")