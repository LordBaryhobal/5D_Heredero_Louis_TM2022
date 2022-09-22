#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates QR Code mask figures

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

import numpy as np
from PIL import Image

MASKS = [
    lambda x, y: (x+y) % 2 == 0,
    lambda x, y: y % 2 == 0,
    lambda x, y: (x) % 3 == 0,
    lambda x, y: (x+y) % 3 == 0,
    lambda x, y: (y//2+x//3) % 2 == 0,
    lambda x, y: ((x*y) % 2 + (x*y) % 3) == 0,
    lambda x, y: ((x*y) % 2 + (x*y) % 3) % 2 == 0,
    lambda x, y: ((x+y) % 2 + (x*y) % 3) % 2 == 0
]

if __name__ == '__main__':
    for i, mask in enumerate(MASKS):
        a = np.ones([21, 21], dtype="uint8")
        for y in range(a.shape[0]):
            for x in range(a.shape[1]):
                if mask(x, y):
                    a[y, x] = 0

        a *= 0xffffff
        img = Image.fromarray(a)
        img.save(f"qr_mask_{i}.png")
