#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates Lycacode data layout matrix figure

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

import pygame
import numpy as np
import hamming

S = 600

class LycacodeError(Exception):
    pass

class Lycacode:
    RES = 3
    BLOCKSIZE = 7

    MODE_PERSON = 0
    MODE_LOC = 1
    MODE_LINK = 2
    MODE_TEXT = 3

    PERSON_STUDENT = 0
    PERSON_TEACHER = 1
    PERSON_OTHER = 2

    BLACK = (158,17,26)
    #BLACK = (0,0,0)
    WHITE = (255,255,255)

    OFFSETS = [(0,-1), (1,0), (0,1), (-1,0)]

    MASKS = [
        lambda x, y: x%3 == 0,
        lambda x, y: y%3 == 0,
        lambda x, y: (x+y)%3 == 0,
        lambda x, y: (x%3)*(y%3)==0,
        lambda x, y: (y//3+x//3)%2==0,
        lambda x, y: (y%3-1)*(x%3-y%3-2)*(y%3-x%3-2)==0,
        lambda x, y: (abs(13-x)+abs(13-y))%3==1,
        lambda x, y: (1-x%2 + max(0, abs(13-y)-abs(13-x))) * (1-y%2 + max(0,abs(13-x)-abs(13-y))) == 0
    ]
    
    MASK = True
    
    def __init__(self, data, mode):
        self.data = data
        self.mode = mode
        self.encode()
        self.create_matrix()

    def encode(self):
        self.bits = f"{self.mode:02b}"

        if self.mode == self.MODE_PERSON:
            type_ = self.data["type"]
            id_ = self.data["id"]
            self.bits += f"{type_:02b}"
            self.bits += f"{id_:020b}"

            if type_ == self.PERSON_STUDENT:
                year = self.data["year"]
                class_ = self.data["class"]
                self.bits += f"{year:03b}"
                self.bits += f"{class_:04b}"
                in1, in2 = self.data["initials"]
                in1, in2 = ord(in1)-ord("A"), ord(in2)-ord("A")
                self.bits += f"{in1:05b}"
                self.bits += f"{in2:05b}"
                # 83 left

            elif type_ == self.PERSON_TEACHER:
                # 100 left
                pass

            elif type_ == self.PERSON_OTHER:
                # 100 left
                pass


        elif self.mode == self.MODE_LOC:
            section = self.data["section"]
            room = self.data["room"]
            self.bits += f"{section:03b}"
            self.bits += f"{room:08b}"
            # 107 left

        elif self.mode == self.MODE_LINK:
            self.bits += f"{self.data:032b}"
            # 86 left

        elif self.mode == self.MODE_TEXT: # max 13 chars
            data = self.data.encode("utf-8")
            self.bits += f"{len(data):07b}"
            self.bits += "".join(list(map(lambda b: f"{b:08b}", data)))
            # 7 left

        else:
            raise LycacodeError(f"Invalid mode {self.mode}")

        ds = self.BLOCKSIZE-self.BLOCKSIZE.bit_length()
        total_bits = (self.RES**2 * 24 - 6)
        data_bits = total_bits * ds // self.BLOCKSIZE
        self.bits += "0"*(ds-len(self.bits)%ds)
        s = ""
        i = 0
        left = data_bits-len(self.bits)
        while len(s) < left:
            s += f"{i:0b}"
            i += 1
        s = s[:left]
        self.bits += s
        self.bits = hamming.encode(self.bits, self.BLOCKSIZE)

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
        mask_area = np.where(self.matrix == -1, 1, 0)
        self.matrix[R*4, R*4+1:R*5-1] = 1
        self.matrix[R*5-1, R*4] = 1
        self.matrix[R*4+1:R*5-1, R*4+1:R*5-1] = 1

        bits = list(map(int, self.bits))
        bits = np.reshape(bits, [-1,self.BLOCKSIZE]).T
        bits = np.reshape(bits, [-1]).tolist()

        for y in range(R*9):
            for x in range(R*9):
                if self.matrix[y,x] == -1:
                    self.matrix[y,x] = bits.pop(0)

                if len(bits) == 0:
                    break

            if len(bits) == 0:
                break
        
        if self.MASK:
            best = [None, None, None]
            for i, mask in enumerate(self.MASKS):
                score, matrix = self.evaluate(mask, mask_area)
                if best[0] is None or score < best[0]:
                    best = (score, matrix, i)

            self.matrix = best[1]
            id_ = list(map(int, f"{best[2]:03b}"))
            self.matrix[0, R*4:R*5] = id_ # mask
            self.matrix[-1, R*4:R*5] = id_ # mask
    
    def evaluate(self, mask, mask_area):
        matrix = self.matrix.copy()
        for y in range(self.matrix.shape[0]):
            for x in range(self.matrix.shape[1]):
                if mask_area[y][x] and mask(x,y):
                    matrix[y][x] = 1-matrix[y][x]

        score = 0
        
        # 3 or more of the same color (horizontal)
        for y in range(self.matrix.shape[0]):
            c = 0
            col = -1
            for x in range(self.matrix.shape[1]):
                if matrix[y][x] == -1: continue
                if col != matrix[y][x]:
                    c = 0
                    col = matrix[y][x]
                c += 1
                if c == 3:
                    score += 4
                elif c > 3:
                    score += 2
        
        # 3 or more of the same color (vertical)
        for x in range(self.matrix.shape[1]):
            c = 0
            col = -1
            for y in range(self.matrix.shape[0]):
                if matrix[y][x] == -1: continue
                if col != matrix[y][x]:
                    c = 0
                    col = matrix[y][x]
                c += 1
                if c == 3:
                    score += 4
                elif c > 3:
                    score += 2
        
        # 2x2 blocks of the same color
        for y in range(matrix.shape[0]-1):
            for x in range(matrix.shape[1]-1):
                if matrix[y][x] == -1: continue
                zone = matrix[y:y+2, x:x+2]
                if np.all(zone == zone[0,0]):
                    score += 2
        
        # more dots/1s => higher score
        total = matrix.shape[0]*matrix.shape[1]
        dots = np.sum(matrix == 1)
        percent = 100*dots//total
        score += percent//5 * 2

        return score, matrix
    
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
        
        pts = [
            (4, 4), (4, 2), (3, 2), (3, 1), (4, 1), (4, 0), (5, 0), (5, 1), (6, 1), (6, 2), (5, 2),
            (5, 4), (7, 4), (7, 3), (8, 3), (8, 4), (9, 4), (9, 5), (8, 5), (8, 6), (7, 6), (7, 5),
            (5, 5), (5, 7), (6, 7), (6, 8), (5, 8), (5, 9), (4, 9), (4, 8), (3, 8), (3, 7), (4, 7),
            (4, 5), (2, 5), (2, 6), (1, 6), (1, 5), (0, 5), (0, 4), (1, 4), (1, 3), (2, 3), (2, 4)
        ]
        pygame.draw.polygon(surf, (0,0,0), [(O+s*p[0], O+s*p[1]) for p in pts], 1)
        

if __name__ == "__main__":
    pygame.init()

    surf = pygame.Surface([S, S])

    code = Lycacode({
        "type": Lycacode.PERSON_STUDENT,
        "id": 16048,
        "year": 5,
        "class": 3,
        "initials": "LH"
        }, Lycacode.MODE_PERSON)

    #code = Lycacode("Embarquement", Lycacode.MODE_TEXT)
    """code = Lycacode({
        "section": 4,
        "room": 209
        }, Lycacode.MODE_LOC)"""
    #code = Lycacode(1, Lycacode.MODE_LINK)
    code.display(surf)
    
    pygame.image.save(surf, "lycacode_data_layout.png")