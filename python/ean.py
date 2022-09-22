#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module can generate EAN-8 and EAN-13 barcodes

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

import pygame

A = [
    0b0001101,
    0b0011001,
    0b0010011,
    0b0111101,
    0b0100011,
    0b0110001,
    0b0101111,
    0b0111011,
    0b0110111,
    0b0001011
]

# XOR 0b1111111
C = list(map(lambda a: a^127, A))

# Reverse bit order
B = list(map(lambda c: int(f"{c:07b}"[::-1], 2), C))

ean13_patterns = [
    "AAAAAA",
    "AABABB",
    "AABBAB",
    "AABBBA",
    "ABAABB",
    "ABBAAB",
    "ABBBAA",
    "ABABAB",
    "ABABBA",
    "ABBABA"
]

def bin_list(n):
    return list(map(int, f"{n:07b}"))

def luhn(digits):
    checksum = sum([
        digits[-i-1]*(3-i%2*2)
        for i in range(len(digits))
    ])
    ctrl_key = 10 - checksum%10
    if ctrl_key == 10:
        ctrl_key = 0

    return ctrl_key

def ean8(digits):
    digits.append(luhn(digits))
    elmts = []

    elmts += [1,0,1] #delimiter
    for digit in digits[:4]:
        elmts += bin_list(A[digit])

    elmts += [0,1,0,1,0] #middle delimiter
    for digit in digits[4:]:
        elmts += bin_list(C[digit])

    elmts += [1,0,1] #delimiter
    return elmts

def ean13(digits):
    pattern = ean13_patterns[digits[0]]
    digits.append(luhn(digits))
    elmts = []

    elmts += [1,0,1] #delimiter
    for d in range(1,7):
        _ = A if pattern[d-1] == "A" else B
        digit = digits[d]
        elmts += bin_list(_[digit])

    elmts += [0,1,0,1,0] #middle delimiter
    for digit in digits[7:]:
        elmts += bin_list(C[digit])

    elmts += [1,0,1] #delimiter
    return elmts

def draw_barcode(barcode, win):
    width = win.get_width()*0.8
    height = win.get_height()*0.5
    bar_w = width/len(barcode)
    rnd_bar_w = round(bar_w)

    win.fill((255,255,255))
    x = win.get_width()*0.1
    y = win.get_height()*0.25

    for c in barcode:
        if c:
            pygame.draw.rect(win, (0,0,0), [x, y, rnd_bar_w, height])

        x += bar_w

if __name__ == "__main__":
    import base
    
    b = base.Base(800, 500, "EAN-8 / EAN-13 barcode generator")
    
    #barcode = ean8([8,4,2,7,3,7,2])
    barcode = ean13([9,7,8,2,9,4,0,6,2,1,0,5])
    
    draw_barcode(barcode, b.w)
    
    b.main()