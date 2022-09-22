#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates lycacode frame dimensions figure

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

import pygame
import numpy as np

S = 600

class Lycacode:
    RES = 3

    BLACK = (158,17,26)
    #BLACK = (0,0,0)
    WHITE = (255,255,255)

    OFFSETS = [(0,-1), (1,0), (0,1), (-1,0)]

    def __init__(self):
        R = self.RES
        self.matrix = np.zeros([R*9, R*9])-1
        self.matrix[R*4:R*5, :] = 0
        self.matrix[:, R*4:R*5] = 0
        self.matrix[R:R*2, R*3:R*6] = 0
        self.matrix[R*3:R*6, R:R*2] = 0
        self.matrix[-R*2:-R, -R*6:-R*3] = 0
        self.matrix[-R*6:-R*3, -R*2:-R] = 0
        
        self.font = pygame.font.SysFont("arial", 30, bold=True)

    def display(self, surf):
        R = self.RES
        S = min(surf.get_size())*2
        s = int(S/12/R)*R
        O = (S-s*9)/2

        surf.fill(self.WHITE)

        # Frame
        pygame.draw.rect(surf, self.BLACK, [O-s, O-s, s*11, s*11])
        pygame.draw.rect(surf, self.WHITE, [O-s*0.5, O-s*0.5, s*10, s*10])

        # Cross
        for i in range(4):
            dx, dy = self.OFFSETS[i]
            X, Y = S/2 + dx*s*3, S/2 + dy*s*3
            for j in range(3):
                dx2, dy2 = self.OFFSETS[(i+j-1)%4]
                pygame.draw.circle(surf, self.BLACK, [X+dx2*s, Y+dy2*s], 0.75*s)

            pygame.draw.rect(surf, self.BLACK, [X-(1.5-abs(dx))*s, Y-(1.5-abs(dy))*s, s*(3-abs(dx)*2), s*(3-abs(dy)*2)])

        pygame.draw.rect(surf, self.BLACK, [O, S/2-s/2, 9*s, s])
        pygame.draw.rect(surf, self.BLACK, [S/2-s/2, O, s, 9*s])

        for y in range(9):
            for x in range(9):
                if self.matrix[y*R, x*R] != -1:
                    pygame.draw.rect(surf, (0,0,0), [O+x*s, O+y*s, s+1, s+1], 2)
        
        col = (17,158,147)
        col = (0,0,0)
        pygame.draw.line(surf, col, [O-s, O+s*3], [O, O+s*3], 3)
        pygame.draw.line(surf, col, [O-s, O+s*3-s/6], [O-s, O+s*3+s/6], 3)
        pygame.draw.line(surf, col, [O, O+s*3-s/6], [O, O+s*3+s/6], 3)
        txt = self.font.render("S", True, col)
        surf.blit(txt, [O-s/2-txt.get_width()/2, O+s*3-s/6-txt.get_height()])
        
        pygame.draw.line(surf, col, [O+s, O+s*3-s/6], [O+s*2, O+s*3-s/6], 3)
        pygame.draw.line(surf, col, [O+s, O+s*3-s/3], [O+s, O+s*3], 3)
        pygame.draw.line(surf, col, [O+s*2, O+s*3-s/3], [O+s*2, O+s*3], 3)
        txt = self.font.render("S", True, col)
        surf.blit(txt, [O+3*s/2-txt.get_width()/2, O+s*3-s/3-txt.get_height()])
        
        pygame.draw.line(surf, col, [O+s, O-s], [O+s, O-s/2], 3)
        pygame.draw.line(surf, col, [O+s-s/6, O-s], [O+s+s/6, O-s], 3)
        pygame.draw.line(surf, col, [O+s-s/6, O-s/2], [O+s+s/6, O-s/2], 3)
        txt = self.font.render("S/2", True, col)
        surf.blit(txt, [O+s-s/6-txt.get_width(), O-s*0.75-txt.get_height()/2])

if __name__ == "__main__":
    pygame.init()
    
    surf = pygame.Surface([S, S])
    
    code = Lycacode()
    code.display(surf)
    pygame.image.save(surf, "lycacode_frame.png")
