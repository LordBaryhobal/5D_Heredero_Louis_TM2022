#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module can be used to scan Lycacodes

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

import cv2
import numpy as np
from math import sqrt
import hamming

DB_WIN = True
TOL_CNT_DIST = 20
R = 3

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
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (5,5), 0)
    #if DB_WIN: cv2.imshow("grey", grey)

    #bw = cv2.threshold(grey, np.mean(grey), 255, cv2.THRESH_BINARY)[1]
    #bw = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 0)
    bw = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
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

        s = 320/10/R
        matrix = np.zeros([R*9, R*9])-1
        matrix[R*4:R*5, 0:] = 0
        matrix[0:, R*4:R*5] = 0
        matrix[R:R*2, R*3:R*6] = 0
        matrix[R*3:R*6, R:R*2] = 0
        matrix[-R*2:-R, -R*6:-R*3] = 0
        matrix[-R*6:-R*3, -R*2:-R] = 0

        dots = warped.copy()
        dots = cv2.cvtColor(dots, cv2.COLOR_GRAY2BGR)
        for y in range(R*9):
            cv2.line(dots, (0, int(s*R/2+(y+1)*s)), (320, int(s*R/2+(y+1)*s)), (0,255,0), 1)
            cv2.line(dots, (int(s*R/2+(y+1)*s), 0), (int(s*R/2+(y+1)*s), 320), (0,255,0), 1)
            for x in range(R*9):
                if matrix[y, x] == 0:
                    X, Y = (x+R/2)*s, (y+R/2)*s
                    val = np.mean(warped[int(Y+s/2)-1:int(Y+s/2)+2, int(X+s/2)-1:int(X+s/2)+2])
                    matrix[y, x] = int(round(val/255))
                    cv2.circle(dots, (int(Y+s/2), int(X+s/2)), 2, (0,0,255), 1)

        if DB_WIN:
            cv2.imshow("dots", dots)

        v = _decode(matrix)
        if not v is None:
            return v

    return None
    #return _decode(matrix)

def _decode(matrix):
    OFFSETS = [[(1,0), (R-1,1)], [(R-1,1), (R,R-1)], [(1,R-1), (R-1,R)], [(0,1), (1,R-1)]]
    I = None
    for i in range(4):
        s, e = OFFSETS[i]
        dx1, dy1 = s
        dx2, dy2 = e
        # Find top
        if (matrix[R*4+dy1:R*4+dy2, R*4+dx1:R*4+dx2] == 1).all():
            I = i
    
    
    if I is None:
        return
    
    # Put top on top
    matrix = np.rot90(matrix, I)
    
    # If left on right, flip
    if matrix[R*5-1, R*5-1] == 1:
        matrix = np.fliplr(matrix)
    
    # If not left on left -> problem
    elif matrix[R*5-1, R*4] != 1:
        return
    
    matrix[R*4:R*5, R*4:R*5] = -1
    
    mask_i = "".join([str(int(b)) for b in matrix[0, R*4:R*5]])
    mask_i = int(mask_i, 2)
    matrix[0, R*4:R*5] = -1
    matrix[-1, R*4:R*5] = -1
    
    for y in range(R*9):
        for x in range(R*9):
            if MASKS[mask_i](x,y) and matrix[y][x] != -1:
                matrix[y][x] = 1-matrix[y][x]
    
    if DB_WIN:
        img = ((matrix+2)%3)/2*255
        cv2.namedWindow("matrix", cv2.WINDOW_NORMAL)
        cv2.imshow("matrix", np.array(img, dtype="uint8"))
    
    bits = []
    for y in range(R*9):
        for x in range(R*9):
            if matrix[y, x] != -1:
                bits.append(int(matrix[y,x]))
    
    bits = np.reshape(bits, [-1, int(len(bits)/7)]).T
    bits = np.reshape(np.array(bits), [-1])
    bits = "".join(list(map(str, bits)))
    
    data, errors = hamming.decode(bits, 7)
    if errors > 6:
        return
    
    mode, data = int(data[:2],2), data[2:]
    
    if mode == 0:
        person = {}
        type_, data = int(data[:2],2), data[2:]
        id_, data = int(data[:20],2), data[20:]
    
        person["type"] = type_
        person["id"] = id_

        # Student
        if type_ == 0:
            year, data = int(data[:3],2), data[3:]
            class_, data = int(data[:4],2), data[4:]
            person["year"] = year
            person["class"] = class_
    
            in1, in2, data = int(data[:5],2), int(data[5:10],2), data[10:]
            in1, in2 = chr(in1+ord("A")), chr(in2+ord("A"))
    
            person["initials"] = in1+in2
    
        # Teacher
        elif type_ == 1:
            pass

        # Other
        elif type_ == 2:
            pass

        else:
            print(f"Invalid person type {type_}")

        data = person

    elif mode == 1:
        loc = {}
        section, data = int(data[:3],2), data[3:]
        room, data = int(data[:9],2), data[9:]

        loc["section"] = section
        loc["room"] = room
        data = loc

    elif mode == 2:
        data = int(data[:32],2)

    elif mode == 3:
        length, data = int(data[:4],2), data[4:]
        if length*8 > len(data): return

        data = bytes([int(data[i*8:i*8+8],2) for i in range(length)])
        try:
            data = data.decode("utf-8")

        except UnicodeDecodeError:
            return

    else:
        #raise LycacodeError(f"Invalid mode {self.mode}")
        print(f"Invalid mode {self.mode}")
        return

    return data

if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    cam = cv2.VideoCapture(0)
    while True:
        ret, img = cam.read()

        data = decode(img)

        if not data is None:
            print(data)

        cv2.imshow("src", img)
        cv2.waitKey(1)
