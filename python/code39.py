#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module can generate Code-39 barcodes

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

import pygame

code39_dict = {
    "A": "100001001", "B": "001001001",
    "C": "101001000", "D": "000011001",
    "E": "100011000", "F": "001011000",
    "G": "000001101", "H": "100001100",
    "I": "001001100", "J": "000011100",
    "K": "100000011", "L": "001000011",
    "M": "101000010", "N": "000010011",
    "O": "100010010", "P": "001010010",
    "Q": "000000111", "R": "100000110",
    "S": "001000110", "T": "000010110",
    "U": "110000001", "V": "011000001",
    "W": "111000000", "X": "010010001",
    "Y": "110010000", "Z": "011010000",
    "0": "000110100", "1": "100100001",
    "2": "001100001", "3": "101100000",
    "4": "000110001", "5": "100110000",
    "6": "001110000", "7": "000100101",
    "8": "100100100", "9": "001100100",
    " ": "011000100", "-": "010000101",
    "$": "010101000", "%": "000101010",
    ".": "110000100", "/": "010100010",
    "+": "010001010", "*": "010010100"
}

def code39(text):
    text = text.upper()
    text = map(lambda c: code39_dict[c], text)
    return "0".join(text)

def draw_barcode(barcode, win):
    barcode = list(map(int, barcode))
    width = win.get_width()*0.8
    height = win.get_height()*0.5
    thicks = sum(barcode)
    thins = len(barcode)-thicks
    bar_w = width/(thicks*2+thins)

    win.fill((255,255,255))
    x = win.get_width()*0.1
    y = win.get_height()*0.25

    for i, c in enumerate(barcode):
        w = 2*bar_w if c else bar_w
        if i%2 == 0:
            pygame.draw.rect(win, (0,0,0), [x, y, w, height])

        x += w

if __name__ == "__main__":
    import base
    
    b = base.Base(800, 500, "Code-39 barcode generator")
    
    barcode = code39("*CODE-39*")
    draw_barcode(barcode, b.w)
    
    b.main()