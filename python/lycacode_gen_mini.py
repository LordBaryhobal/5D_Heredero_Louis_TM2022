#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module can be used to create and display Mini-Lycacodes

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

import pygame
import numpy as np
import hamming

S = 600

class LycacodeError(Exception):
    pass

class LycacodeMini:
    BLACK = (158,17,26)
    #BLACK = (0,0,0)
    WHITE = (255,255,255)
    
    OFFSETS = [(0,-1), (1,0), (0,1), (-1,0)]
    
    FRAME = True
    CIRCLES = True
    DOTS = True
    DB_SQUARES = False

    def __init__(self, id_):
        self.id = id_
        self.encode()
        self.create_matrix()

    def encode(self):
        self.bits = f"{self.id:020b}"

        self.bits = list(map(int, self.bits))
        parity = [sum(self.bits[i*5:i*5+5])%2 for i in range(4)]
        for i in range(4):
            self.bits.insert((4-i)*5, parity[3-i])


    def create_matrix(self):
        self.matrix = np.zeros([9, 9])-1
        self.matrix[4:5, :] = 0
        self.matrix[:, 4:5] = 0
        self.matrix[1:2, 3:6] = 0
        self.matrix[3:6, 1:2] = 0
        self.matrix[-2:-1, -6:-3] = 0
        self.matrix[-6:-3, -2:-1] = 0
        self.matrix[4,4] = -1

        for y in range(9):
            for x in range(9):
                if self.matrix[y,x] == 0:
                    self.matrix[y,x] = self.bits.pop(0)

                if len(self.bits) == 0:
                    break

            if len(self.bits) == 0:
                break

    def display(self, surf):
        S = min(surf.get_size())
        s = int(S/12/3)*3
        O = (S-s*9)/2

        surf.fill(self.WHITE)

        # Frame
        if self.FRAME:
            pygame.draw.rect(surf, self.BLACK, [O-s, O-s, s*11, s*11])
            pygame.draw.rect(surf, self.WHITE, [O-s*0.5, O-s*0.5, s*10, s*10])

        # Cross
        for i in range(4):
            dx, dy = self.OFFSETS[i]
            X, Y = S/2 + dx*s*3, S/2 + dy*s*3
            if self.CIRCLES:
                for j in range(3):
                    dx2, dy2 = self.OFFSETS[(i+j-1)%4]
                    pygame.draw.circle(surf, self.BLACK, [X+dx2*s, Y+dy2*s], 0.75*s)

            pygame.draw.rect(surf, self.BLACK, [X-(1.5-abs(dx))*s, Y-(1.5-abs(dy))*s, s*(3-abs(dx)*2), s*(3-abs(dy)*2)])

        pygame.draw.rect(surf, self.BLACK, [O, S/2-s/2, 9*s, s])
        pygame.draw.rect(surf, self.BLACK, [S/2-s/2, O, s, 9*s])

        # Dots
        if self.DOTS:
            for y in range(9):
                for x in range(9):
                    if self.matrix[y, x] == 1:
                        pygame.draw.circle(surf, self.WHITE, [O+(x+0.5)*s, O+(y+0.5)*s], s/3)

        # Center
        pygame.draw.circle(surf, self.WHITE, [O+4.5*s, O+4.25*s], s/6)
        pygame.draw.circle(surf, self.WHITE, [O+4.25*s, O+4.75*s], s/6)
    
    def save(self, path):
        S = 600
        s = int(S/12)
        O = (S-s*9)/2
        
        BLACK = "#9E111A"
        WHITE = "#FFFFFF"
        
        with open(path, "w") as f:
            f.write(f"<svg width='{S}px' height='{S}px' viewbox='0 0 {S} {S}'>\n")
            # Background
            f.write(f"<rect x='0' y='0' width='{S}' height='{S}' fill='{WHITE}' />\n")
            
            # Frame
            f.write(f"<rect x='{O-s*0.75}' y='{O-s*0.75}' width='{s*10.5}' height='{s*10.5}' style='fill:none;stroke:{BLACK};stroke-width:{0.5*s};' />\n")
            
            # Cross
            for i in range(4):
                dx, dy = self.OFFSETS[i]
                X, Y = S/2 + dx*s*3, S/2 + dy*s*3
                if self.CIRCLES:
                    for j in range(3):
                        dx2, dy2 = self.OFFSETS[(i+j-1)%4]
                        f.write(f"<circle cx='{X+dx2*s}' cy='{Y+dy2*s}' r='{0.75*s}' style='fill:{BLACK};stroke:none;stroke-width:0;' />\n")
                
                f.write(f"<rect x='{X-(1.5-abs(dx))*s}' y='{Y-(1.5-abs(dy))*s}' width='{s*(3-abs(dx)*2)}' height='{s*(3-abs(dy)*2)}' style='fill:{BLACK};stroke:none;stroke-width:0;' />\n")
            
            # Cross
            f.write(f"<rect x='{O}' y='{S/2-s/2}' width='{9*s}' height='{s}' style='fill:{BLACK};stroke:none;stroke-width:0;' />\n")
            f.write(f"<rect x='{S/2-s/2}' y='{O}' width='{s}' height='{9*s}' style='fill:{BLACK};stroke:none;stroke-width:0;' />\n")
            
            # Dots
            if self.DOTS:
                for y in range(9):
                    for x in range(9):
                        if self.matrix[y, x] == 1:
                            f.write(f"<circle cx='{O+(x+0.5)*s}' cy='{O+(y+0.5)*s}' r='{s/3}' style='fill:{WHITE};stroke:none;stroke-width:0;' />\n")
            
            # Center
            f.write(f"<circle cx='{O+4.5*s}' cy='{O+4.25*s}' r='{s/6}' style='fill:{WHITE};stroke:none;stroke-width:0;' />\n")
            f.write(f"<circle cx='{O+4.25*s}' cy='{O+4.75*s}' r='{s/6}' style='fill:{WHITE};stroke:none;stroke-width:0;' />\n")
            
            f.write("</svg>")

def save(self):
    path = input("Save as (.png or .svg): ")
    
    if path.endswith(".svg"):
        code.save(path)
    
    else:
        pygame.image.save(w, path)

if __name__ == "__main__":
    import base
    
    b = base.Base(S, S, "Mini-Lycacode generator")
    
    code = LycacodeMini(16048)
    code.display(b.w)
    
    b.save = save
    b.main()