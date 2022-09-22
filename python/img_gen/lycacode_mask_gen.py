#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates Lycacode mask figures

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

import numpy as np
from PIL import Image

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

if __name__ == '__main__':
    R = 3
    for i, mask in enumerate(MASKS):
        a = np.ones([27, 27], dtype="uint8")
        a[R*4:R*5, :] = 2
        a[:, R*4:R*5] = 2
        a[R:R*2, R*3:R*6] = 2
        a[R*3:R*6, R:R*2] = 2
        a[-R*2:-R, -R*6:-R*3] = 2
        a[-R*6:-R*3, -R*2:-R] = 2
        
        a[R*4:R*5, R*4:R*5] = 1
        
        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                if mask(x, y) and a[y,x] == 2:
                    a[y, x] = 0

        a *= 0x7f
        img = Image.fromarray(a)
        img.save(f"lycacode_mask_{i}.png")
