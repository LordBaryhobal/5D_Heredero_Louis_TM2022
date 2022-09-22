#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module can be used to scan Mini-Lycacodes

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

import cv2
import numpy as np
from math import sqrt
import hamming

DB_WIN = False
TOL_CNT_DIST = 20

def center(c):
    M = cv2.moments(c)

    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    return (None, None)

def dist(p1, p2):
    return sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def is_symbol(i, cnts, hrcy):
    c1 = cnts[i]
    h1 = hrcy[0][i]
    cX1, cY1 = center(c1)
    if len(c1) != 4:
        return False

    if cX1 is None:
        return False

    if h1[2] == -1:
        return False

    i2 = h1[2]
    c2 = cnts[i2]
    h2 = hrcy[0][i2]
    cX2, cY2 = center(c2)
    if cX2 is None:
        return False

    if len(c2) != 8:
        return False

    if abs(dist((cX1, cY1), (cX2, cY2))) > TOL_CNT_DIST:
        return False

    return True

def decode(img):
    #grey = img[:,:,2]
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #grey = cv2.GaussianBlur(grey, (5,5), 0)
    #if DB_WIN: cv2.imshow("grey", grey)

    #bw = cv2.threshold(grey, np.mean(grey), 255, cv2.THRESH_BINARY)[1]
    bw = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY)[1]
    #bw = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
    #bw = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    if DB_WIN: cv2.imshow("bw", bw)

    #laplacian = cv2.Laplacian(bw, cv2.CV_8U, 15)
    #cv2.imshow("laplacian", laplacian)

    contours, hierarchy = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if DB_WIN:
        img2 = img.copy()
        cv2.drawContours(img2, contours, -1, (0,255,0), 1)

    candidates = []

    contours = list(contours)
    for i, cnt in enumerate(contours):
        peri = cv2.arcLength(cnt, True)
        contours[i] = cv2.approxPolyDP(cnt, 0.04 * peri, True)

    for i in range(len(contours)):
        if is_symbol(i, contours, hierarchy):
            candidates.append(i)

    if DB_WIN:
        for i in candidates:
            cv2.drawContours(img2, contours, i, (0,0,255), 1)
            cv2.drawContours(img2, contours, hierarchy[0][i][2], (0,0,255), 1)

        cv2.imshow("contours", img2)

    if DB_WIN:
        img3 = img.copy()
        cv2.drawContours(img3, contours, -1, (0,0,255), 1)
        cv2.imshow("contours-all", img3)

    if len(candidates) == 0:
        return

    for i in candidates:
        i = candidates[0]
        j = hierarchy[0][i][2]
        cnt1, cnt2 = contours[i][::-1], contours[j]

        from_ = [ cnt1[0], cnt1[1], cnt1[2], cnt1[3] ]
        to = [ (0,0), (320,0), (320,320), (0,320) ]

        M = cv2.getPerspectiveTransform(np.array(from_, dtype="float32"), np.array(to, dtype="float32"))
        #_ = cv2.dilate(bw, (11,11))
        #warped = cv2.warpPerspective(_, M, (320,320))
        warped = cv2.warpPerspective(bw, M, (320,320))


        if DB_WIN:
            cv2.imshow("warped", warped)

        s = 320/10
        matrix = np.zeros([9, 9])-1
        matrix[4:5, 0:] = 0
        matrix[0:, 4:5] = 0
        matrix[1:2, 3:6] = 0
        matrix[3:6, 1:2] = 0
        matrix[-2:-1, -6:-3] = 0
        matrix[-6:-3, -2:-1] = 0

        dots = warped.copy()
        dots = cv2.cvtColor(dots, cv2.COLOR_GRAY2BGR)
        for y in range(9):
            cv2.line(dots, (0, int(s/2+(y+1)*s)), (320, int(s/2+(y+1)*s)), (0,255,0), 1)
            cv2.line(dots, (int(s/2+(y+1)*s), 0), (int(s/2+(y+1)*s), 320), (0,255,0), 1)
            for x in range(9):
                if matrix[y, x] == 0:
                    X, Y = (x+0.5)*s, (y+0.5)*s
                    val = np.mean(warped[int(Y+s/2)-1:int(Y+s/2)+2, int(X+s/2)-1:int(X+s/2)+2])
                    matrix[y, x] = int(round(val/255))
                    cv2.circle(dots, (int(Y+s/2), int(X+s/2)), 2, (0,0,255), 1)

        OFFSETS = [(0,-1),(1,0),(0,1),(-1,0)]
        I = None
        for i in range(4):
            dx, dy = OFFSETS[i]
            X, Y = 320/2+dx*s/3, 320/2+dy*s/3
            cv2.circle(dots, (int(Y), int(X)), 2, (0,255,255), 1)
            if np.mean(warped[int(Y)-1:int(Y)+2, int(X)-1:int(X)+2]) > 127:
                I = i

        if I is None:
            continue
        matrix = np.rot90(matrix, I)

        dx, dy = [(1,1), (-1, 1), (-1,-1), (1,-1)][I]
        X, Y = 320/2+dx*s/3, 320/2+dy*s/3

        if np.mean(warped[int(Y)-1:int(Y)+2, int(X)-1:int(X)+2]) > 127:
            matrix = np.fliplr(matrix)

        if DB_WIN:
            cv2.imshow("dots", dots)

        v = _decode(matrix)
        if not v is None:
            return v

    return None
    #return _decode(matrix)

def _decode(matrix):
    matrix[4:5, 4:5] = -1
    
    if DB_WIN:
        img = ((matrix+2)%3)/2*255
        cv2.namedWindow("matrix", cv2.WINDOW_NORMAL)
        cv2.imshow("matrix", np.array(img, dtype="uint8"))
    
    bits = []
    for y in range(9):
        for x in range(9):
            if matrix[y, x] != -1:
                bits.append(int(matrix[y,x]))
    
    for i in range(4):
        if sum(bits[i*6:i*6+6])%2:
            return
    
    data = "".join(list(map(str, bits)))
    
    id_ = int(data[0:5]+data[6:11]+data[12:17]+data[18:23],2)
    
    return id_

if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()

        data = decode(img)

        if not data is None:
            print(data)

        cv2.imshow("src", img)
        cv2.waitKey(10)
