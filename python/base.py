#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module provides a base class to display codes and enable saving

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

import pygame

class Base:
    def __init__(self, width, height, caption):
        pygame.init()
        
        pygame.display.set_caption(caption)
        self.w = pygame.display.set_mode([width, height])
        
        self.controls([
            "CTRL + S: save as",
            "ESC: quit"
        ])
    
    def controls(self, controls, margin=2):
        longest = max(list(map(len, controls))+[10])
        print("┌─" + "─"*(longest+margin) + "─┐")
        
        _ = "\x1b[1;4mControls:\x1b[0m"
        _ += " "*(longest+margin-9)
        print(f"│ " + _ + " │")
        for c in controls:
            print("│ " + " "*margin + c.ljust(longest) + " │")
        print("└─" + "─"*(longest+margin) + "─┘")
    
    def main(self):
        pygame.display.flip()
        
        stop = False
        while not stop:
            event = pygame.event.wait()
            # ESC or close button -> quit
            if event.type == pygame.QUIT:
                stop = True

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    stop = True

                # CTRL+S -> save image
                elif event.key == pygame.K_s and \
                     event.mod & pygame.KMOD_CTRL:
                    self.save()
    
    def save(self):
        path = input("Save as: ")
        pygame.image.save(self.w, path)