#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates QR Code mask evaluation figures

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

import pygame
import numpy as np

matrix = np.array([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
matrix = np.where(matrix == -0.5, 0, matrix)

class History:
    def __init__(self):
        self.widths = [0]*7
        self.widths[-1] = 4
        self.color = 0
    
    def add(self, col):
        a, b, w = False, False, self.widths.copy()
        if col != self.color:
            self.color = col
            a,b,w = self.check()
            self.widths.pop(0)
            self.widths.append(0)
        
        self.widths[-1] += 1
        return (a, b, w)
    
    def check(self):
        n = self.widths[1]
        
        # if 1:1:3:1:1
        if n > 0 and self.widths[2] == n and self.widths[3] == n*3 and self.widths[4] == n and self.widths[5] == n:
            # add 40 if 4:1:1:3:1:1 + add 40 if 1:1:3:1:1:4
            return (self.widths[0] >= 4, self.widths[6] >= 4, self.widths.copy())
        
        return (False, False, self.widths.copy())
    
    def final(self):
        for i in range(4):
            self.add(0)
        
        return self.check()

def generate_imgs():
    m = ((matrix.copy()+2)%3)*127
    mat = np.ones((m.shape[0]+8, m.shape[1]+8))*255
    mat[4:-4, 4:-4] = m
    
    size = 30
    surf = pygame.Surface([mat.shape[0]*size, mat.shape[0]*size])
    
    surf.fill((255,255,255))
    
    for y in range(mat.shape[0]):
        for x in range(mat.shape[1]):
            col = mat[y, x]
            col = (col, col, col)
            pygame.draw.rect(surf, col, [x*size, y*size, size, size])
    
    img1, img2, img3, img4 = [surf.copy() for i in range(4)]
    f = pygame.font.SysFont("sans", 20)
    f2 = pygame.font.SysFont("sans", 48, bold=True)
    f3 = pygame.font.SysFont("sans", 32, bold=True)
    
    #Condition 1 (horizontal)
    counts = []
    scores = []
    for y in range(matrix.shape[0]):
        col = -1
        row = []
        s = 0
        for x in range(matrix.shape[1]):
            if matrix[y,x] != col:
                if len(row) > 0 and row[-1] < 5:
                    n = row[-1]
                    row = row[:-n]+[0]*n
                
                row.append(1)
                col = matrix[y,x]
            
            else:
                row.append(row[-1]+1)
                if row[-1] == 5:
                    s += 3
                elif row[-1] > 5:
                    s += 1
            
        counts.append(row)
        scores.append(s)
    
    for i, c in enumerate(counts[0]):
        if c:
            txt = f.render(str(c), True, (255,255,255))
            img1.blit(txt, [(i+4.5)*size-txt.get_width()/2, 4.5*size-txt.get_height()/2])
    
    for i, s in enumerate(scores):
        txt = f.render(str(s), True, (255,0,0))
        img1.blit(txt, [size*(5+matrix.shape[1])-txt.get_width()/2, (i+4.5)*size-txt.get_height()/2])
    
    #Condition 1 (vertical)
    scores2 = []
    for x in range(matrix.shape[1]):
        col, count = -1, 0
        s = 0
        for y in range(matrix.shape[0]):
            if matrix[y,x] != col:
                count = 0
                col = matrix[y,x]
            count += 1

            if count == 5:
                s += 3
            elif count > 5:
                s += 1
        
        scores2.append(s)
    
    for i, s in enumerate(scores2):
        txt = f.render(str(s), True, (255,0,0))
        img1.blit(txt, [(i+4.5)*size-txt.get_width()/2, size*(5+matrix.shape[0])-txt.get_height()/2])
    
    txt = f2.render(f"= {sum(scores)+sum(scores2)}", True, (255,0,0))
    img1.blit(txt, [size*(5+matrix.shape[1])-txt.get_width()/2, size*(6+matrix.shape[1])-txt.get_height()/2])
    
    pygame.image.save(img1, "qr_mask_ex_eval_1.png")
    
    #Condition 2
    score = 0
    txtR = f.render("3", True, (255,0,0))
    txtW = f.render("3", True, (255,255,255))
    for y in range(matrix.shape[0]-1):
        for x in range(matrix.shape[1]-1):
            zone = matrix[y:y+2, x:x+2]
            if np.all(zone == zone[0,0]):
                score += 3
                txt = [txtR, txtW][int(zone[0,0])]
                img2.blit(txt, [(x+5)*size-txt.get_width()/2, (y+5)*size-txt.get_height()/2])
    
    txt = f2.render(f"= {score}", True, (255,0,0))
    img2.blit(txt, [size*(5+matrix.shape[1])-txt.get_width()/2, size*(6+matrix.shape[1])-txt.get_height()/2])
    
    pygame.image.save(img2, "qr_mask_ex_eval_2.png")
    
    i = 0
    cols = [(255,0,0),(44,219,99),(26,135,240),(229,205,44)]
    cols = []
    for j in range(36):
        c = pygame.Color(0)
        hsla = [j*40%360, 60, 50, 100]
        c.hsla = hsla
        cols.append(c)
    
    for y in range(matrix.shape[0]):
        hist = History()
        for x in range(matrix.shape[1]):
            a,b,w = hist.add(matrix[y,x])
            
            if a:
                col = cols[min(len(cols)-1,i)]
                X = x-sum(w[1:]) #+4-4
                draw_line(img3, col, size, X, y)
                i += 1
            
            if b:
                col = cols[min(len(cols)-1,i)]
                X = x-sum(w[1:])+4
                draw_line(img3, col, size, X, y)
                i += 1
        
        a,b,w = hist.final()
        if a:
            col = cols[min(len(cols)-1,i)]
            X = matrix.shape[1]-sum(w[1:])
            draw_line(img3, col, size, X, y)
            i += 1
        
        if b:
            col = cols[min(len(cols)-1,i)]
            X = matrix.shape[1]+4-sum(w[1:-1])
            draw_line(img3, col, size, X, y)
            i += 1
    
    for x in range(matrix.shape[1]):
        hist = History()
        for y in range(matrix.shape[0]):
            a,b,w = hist.add(matrix[y,x])
            
            if a:
                col = cols[min(len(cols)-1,i)]
                Y = y-sum(w[1:])
                draw_line(img3, col, size, x, Y, True)
                i += 1
            
            if b:
                col = cols[min(len(cols)-1,i)]
                Y = y-sum(w[1:])+4
                draw_line(img3, col, size, x, Y, True)
                i += 1
        
        a,b,w = hist.final()
        if a:
            col = cols[min(len(cols)-1,i)]
            Y = matrix.shape[0]-sum(w[1:])
            draw_line(img3, col, size, x, Y, True)
            i += 1
        
        if b:
            col = cols[min(len(cols)-1,i)]
            Y = matrix.shape[0]+4-sum(w[1:-1])
            draw_line(img3, col, size, x, Y, True)
            i += 1
    
    txt = f2.render(f"= {i}*40 = {i*40}", True, (255,0,0))
    img3.blit(txt, [size*(4+matrix.shape[1]/2)-txt.get_width()/2, size*(6+matrix.shape[1])-txt.get_height()/2])
    
    pygame.image.save(img3, "qr_mask_ex_eval_3.png")
    
    #Condition 4
    total = matrix.shape[0]*matrix.shape[1]
    dark = np.sum(matrix == 1)
    percent = 100*dark//total
    p1 = percent-(percent%5)
    p2 = p1+5
    p1, p2 = abs(p1-50)/5, abs(p2-50)/5
    score = min(p1,p2)*10
    
    txt = f3.render(f"P = {percent}%", True, (255,0,0))
    img4.blit(txt, [size, size])
    
    txt = f3.render(f"P1 = {percent-(percent%5)}% / P2 = {percent-(percent%5)+5}%", True, (255,0,0))
    img4.blit(txt, [surf.get_width() - size - txt.get_width(), size])
    
    txt = f2.render(f"S = min({int(p1)}, {int(p2)})*10 = {int(score)}", True, (255,0,0))
    img4.blit(txt, [size*(4+matrix.shape[1]/2)-txt.get_width()/2, size*(6+matrix.shape[1])-txt.get_height()/2])
    
    pygame.image.save(img4, "qr_mask_ex_eval_4.png")

def draw_line(surf, col, size, x, y, vert=False):
    a, b = [(x+0.25)*size, (4.5+y)*size], [(x+10.75)*size, (4.5+y)*size]
    if vert:
        a, b = [(4.5+x)*size, (y+0.25)*size], [(4.5+x)*size, (y+10.75)*size]
    
    dx, dy = [size/3, 0] if vert else [0, size/3]
    
    pygame.draw.line(surf, col, a, b, 6)
    pygame.draw.line(surf, col, [a[0]-dx, a[1]-dy], [a[0]+dx, a[1]+dy], 6)
    pygame.draw.line(surf, col, [b[0]-dx, b[1]-dy], [b[0]+dx, b[1]+dy], 6)

if __name__ == "__main__":
    pygame.init()
    
    generate_imgs()