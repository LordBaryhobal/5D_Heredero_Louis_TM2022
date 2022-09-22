#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module can be used to scan QR-Codes

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

import cv2
import numpy as np
from math import sqrt, degrees, atan2, radians, cos, sin
import imutils

DB_WIN = False

TOL_CNT_DIST = 10  # Tolerance distance between finders' centers
MASKS = [
    lambda x,y: (x+y)%2 == 0,
    lambda x,y: y%2 == 0,
    lambda x,y: (x)%3 == 0,
    lambda x,y: (x+y)%3 == 0,
    lambda x,y: (y//2+x//3)%2 == 0,
    lambda x,y: ((x*y)%2 + (x*y)%3) == 0,
    lambda x,y: ((x*y)%2 + (x*y)%3)%2 == 0,
    lambda x,y: ((x+y)%2 + (x*y)%3)%2 == 0
]

ALIGNMENT_PATTERN_LOCATIONS = [
    [],
    [6, 18],
    [6, 22],
    [6, 26],
    [6, 30],
    [6, 34],
    [6, 22, 38],
    [6, 24, 42],
    [6, 26, 46],
    [6, 28, 50],
    [6, 30, 54],
    [6, 32, 58],
    [6, 34, 62],
    [6, 26, 46, 66],
    [6, 26, 48, 70],
    [6, 26, 50, 74],
    [6, 30, 54, 78],
    [6, 30, 56, 82],
    [6, 30, 58, 86],
    [6, 34, 62, 90],
    [6, 28, 50, 72, 94],
    [6, 26, 50, 74, 98],
    [6, 30, 54, 78, 102],
    [6, 28, 54, 80, 106],
    [6, 32, 58, 84, 110],
    [6, 30, 58, 86, 114],
    [6, 34, 62, 90, 118],
    [6, 26, 50, 74, 98, 122],
    [6, 30, 54, 78, 102, 126],
    [6, 26, 52, 78, 104, 130],
    [6, 30, 56, 82, 108, 134],
    [6, 34, 60, 86, 112, 138],
    [6, 30, 58, 86, 114, 142],
    [6, 34, 62, 90, 118, 146],
    [6, 30, 54, 78, 102, 126, 150],
    [6, 24, 50, 76, 102, 128, 154],
    [6, 28, 54, 80, 106, 132, 158],
    [6, 32, 58, 84, 110, 136, 162],
    [6, 26, 54, 82, 110, 138, 166],
    [6, 30, 58, 86, 114, 142, 170]
]

VERSIONS = []
with open("qr_versions.txt", "r") as f:
    versions = f.read().split("\n\n")
    for v in versions:
        lvls = [list(map(int, lvl.split("\t"))) for lvl in v.split("\n")]
        VERSIONS.append(lvls)
VERSIONS = np.array(VERSIONS)

ERROR_CORRECTION =  []
with open("error_correction.txt", "r") as f:
    ecs = f.read().split("\n\n")
    for ec in ecs:
        lvls = [list(map(int, lvl.split("\t"))) for lvl in ec.split("\n")]
        lvls = [lvl + [0]*(6-len(lvl)) for lvl in lvls]
        
        ERROR_CORRECTION.append(lvls)
ERROR_CORRECTION = np.array(ERROR_CORRECTION)

EC_PARAMS =  []
with open("ec_params.txt", "r") as f:
    ecs = f.read().split("\n\n")
    for ec in ecs:
        lvls = [list(map(int, lvl.split("\t"))) for lvl in ec.split("\n") if lvl]
        EC_PARAMS.append(lvls)
EC_PARAMS = np.array(EC_PARAMS)

ALPHANUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"

class GF:
    def __init__(self, val):
        self.val = val

    def copy(self):
        return GF(self.val)

    def __add__(self, n):
        return GF(self.val ^ n.val)

    def __sub__(self, n):
        return GF(self.val ^ n.val)

    def __mul__(self, n):
        if self.val == 0 or n.val == 0:
            return GF(0)

        return GF.EXP[GF.LOG[self.val].val + GF.LOG[n.val].val].copy()

    def __truediv__(self, n):
        if n.val == 0:
            raise ZeroDivisionError
        if self.val == 0:
            return GF(0)

        return GF.EXP[(GF.LOG[self.val].val + 255 - GF.LOG[n.val].val)%255].copy()

    def __pow__(self, n):
        return GF.EXP[(GF.LOG[self.val].val * n.val)%255].copy()

    def __repr__(self):
        return self.val.__repr__()
    
    def log(self):
        return GF.LOG[self.val]

GF.EXP = [GF(0)]*512
GF.LOG = [GF(0)]*256
value = 1
for exponent in range(255):
    GF.LOG[value] = GF(exponent)
    GF.EXP[exponent] = GF(value)
    value = ((value << 1) ^ 285) if value > 127 else value << 1

for i in range(255, 512):
    GF.EXP[i] = GF.EXP[i-255].copy()


class Poly:
    def __init__(self, coefs):
        self.coefs = coefs.copy()

    @property
    def deg(self):
        return len(self.coefs)

    def copy(self):
        return Poly(self.coefs)

    def __add__(self, p):
        d1, d2 = self.deg, p.deg
        deg = max(d1,d2)
        result = [GF(0) for i in range(deg)]
        
        for i in range(d1):
            result[i + deg - d1] = self.coefs[i]

        for i in range(d2):
            result[i + deg - d2] += p.coefs[i]
        
        return Poly(result)

    def __mul__(self, p):
        result = [GF(0) for i in range(self.deg+p.deg-1)]

        for i in range(p.deg):
            for j in range(self.deg):
                result[i+j] += self.coefs[j] * p.coefs[i]

        return Poly(result)

    def __truediv__(self, p):
        dividend = self.coefs.copy()
        dividend += [GF(0) for i in range(p.deg-1)]
        quotient = []

        for i in range(self.deg):
            coef = dividend[i] / p.coefs[0]
            quotient.append(coef)

            for j in range(p.deg):
                dividend[i+j] -= p.coefs[j] * coef

        while dividend[0].val == 0:
            dividend.pop(0)

        return [Poly(quotient), Poly(dividend)]

    def __repr__(self):
        return f"<Poly {self.coefs}>"
    
    def eval(self, x):
        y = GF(0)
        
        for i in range(self.deg):
            y += self.coefs[i] * x**GF(self.deg-i-1)
        
        return y
    
    def del_lead_zeros(self):
        while len(self.coefs) > 1 and self.coefs[0].val == 0:
            self.coefs.pop(0)
        
        if len(self.coefs) == 0:
            self.coefs = [GF(0)]
        
        return self

def center(c):
    M = cv2.moments(c)
    
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    
    return (None, None)

def dist(p1, p2):
    return sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def rotate(o, p, a):
    ox, oy = o
    px, py = p
    a = radians(a)
    c, s = cos(a), sin(a)
    
    return [
        int(ox + c*(px-ox) - s * (py-oy)),
        int(oy + s*(px-ox) + c * (py-oy))
    ]

def rotate_cnt(cnt, o, a):
    c = cnt
    pts = c[:, 0, :]
    pts = np.array([rotate(o, p, a) for p in pts])
    c[:, 0, :] = pts
    
    return c.astype(np.int32)

def is_finder(i, cnts, hrcy):
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
    
    if len(c2) != 4:
        return False
    
    if abs(dist((cX1, cY1), (cX2, cY2))) > TOL_CNT_DIST:
        return False
    
    if h2[2] == -1:
        return False
    
    i3 = h2[2]
    c3 = cnts[i3]
    h3 = hrcy[0][i3]
    cX3, cY3 = center(c3)
    
    if len(c3) != 4:
        return False
    
    if cX3 is None:
        return False
    
    if abs(dist((cX1, cY1), (cX3, cY3))) > TOL_CNT_DIST:
        return False
    
    return True

def decode(img):
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (5,5), 0)
    #if DB_WIN: cv2.imshow("grey", grey)

    bw = cv2.threshold(grey, np.mean(grey), 255, cv2.THRESH_BINARY)[1]
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
        if is_finder(i, contours, hierarchy):
            candidates.append(i)
    
    if DB_WIN: 
        for i in candidates:
            cv2.drawContours(img2, contours, i, (0,0,255), 1)
    
        cv2.imshow("contours", img2)
    
    if DB_WIN: img3 = img.copy()
    corners = []
    corners_cnts = []
    for i1 in candidates:
        i2 = hierarchy[0][i1][2]
        i3 = hierarchy[0][i2][2]
        c1 = contours[i1]
        c2 = contours[i2]
        c3 = contours[i3]
        
        x1, y1 = center(c1)
        x2, y2 = center(c2)
        x3, y3 = center(c3)
        x, y = (x1+x2+x3)/3, (y1+y2+y3)/3
        x, y = int(x), int(y)
        corners.append((x,y))
        corners_cnts.append([c1,c2,c3])
        
        if DB_WIN: 
            cv2.line(img3, [x, y-10], [x, y+10], (0,255,0), 1)
            cv2.line(img3, [x-10, y], [x+10, y], (0,255,0), 1)
            #cv2.drawContours(img3, [c1], 0, (255,0,0), 1)
            #cv2.drawContours(img3, [c2], 0, (0,255,0), 1)
            #cv2.drawContours(img3, [c3], 0, (0,0,255), 1)
    
    if DB_WIN: cv2.imshow("lines", img3)
    
    if len(corners) != 3:
        return
    
    d01 = dist(corners[0], corners[1])
    d02 = dist(corners[0], corners[2])
    d12 = dist(corners[1], corners[2])
    diffs = [abs(d01-d02), abs(d01-d12), abs(d02-d12)]
    mdiff = min(diffs)
    i = diffs.index(mdiff)
    
    a = corners.pop(i)
    b, c = corners
    
    V = [img.shape[1], img.shape[0]]
    d = [(b[0]+c[0])/2, (b[1]+c[1])/2]
    v = [d[0]-a[0], d[1]-a[1]]
    
    C = [img.shape[1]/2, img.shape[0]/2]
    
    angle = degrees(atan2(v[1], v[0]))
    angle_diff = angle-45
    
    if DB_WIN: 
        img4 = img.copy()
        cv2.line(img4, a, [int(d[0]), int(d[1])], (0,255,0), 1)
        cv2.imshow("vecs", img4)
    
    cA = corners_cnts[i][0]
    cA = rotate_cnt(cA, C, -angle_diff)
    rA = cv2.boundingRect(cA)
    
    cA2 = corners_cnts[i][1]
    cA2 = rotate_cnt(cA2, C, -angle_diff)
    rA2 = cv2.boundingRect(cA2)
    
    cB = corners_cnts[i-1][0]
    cB = rotate_cnt(cB, C, -angle_diff)
    rB = cv2.boundingRect(cB)
    
    cC = corners_cnts[i-2][0]
    cC = rotate_cnt(cC, C, -angle_diff)
    rC = cv2.boundingRect(cC)
    
    a, b, c = rotate(C, a, -angle_diff), rotate(C, b, -angle_diff), rotate(C, c, -angle_diff)
    
    if DB_WIN: 
        img5 = img.copy()
        img5 = imutils.rotate(img5, angle_diff)
        cv2.rectangle(img5, rA, (255,0,0), 1)
        cv2.rectangle(img5, rB, (0,255,0), 1)
        cv2.rectangle(img5, rC, (0,0,255), 1)
        cv2.line(img5, a, b, (255,255,255), 1)
        cv2.line(img5, a, c, (255,255,255), 1)
        cv2.imshow("rot", img5)
    
    wul = rA[2]
    if rB[1] < rC[1]:
        wur = rB[2]
    else:
        wur = rC[2]
    
    if rB[1] < rC[1]:
        D = dist(a, b)
    else:
        D = dist(a, c)
    
    X = (wul + wur)/14
    V = (D/X - 10)/4
    V = round(V)
    
    size = V*4+17
    grid = np.zeros([size, size])
    bw_rot = imutils.rotate(bw, angle_diff)
    if DB_WIN: 
        img6 = img.copy()
        img6 = imutils.rotate(img6, angle_diff)
    
    OX, OY = (rA[0]+rA2[0])/2, (rA[1]+rA2[1])/2
    
    #Not fully visible
    if (OX + size*X+1 >= bw_rot.shape[1]) or (OY + size*X+1 >= bw_rot.shape[0]):
        return None
    
    for y in range(size):
        for x in range(size):
            zone = bw_rot[
                int(OY + y*X-1): int(OY + y*X+2),
                int(OX + x*X-1): int(OX + x*X+2)
            ]
            grid[y, x] = 1-round(np.mean(zone)/255)
            #cv2.circle(img6, [int(OX+x*X), int(OY+y*X)], 3, (0,0,255), 1)
    
    #print(size)
    #cv2.rectangle(img6, [int(OX-X/2), int(OY-X/2), int(X), int(X)], (0,255,0), 1)
    #cv2.imshow("grid", img6)
    if DB_WIN: 
        cv2.namedWindow("bw_rot", cv2.WINDOW_NORMAL)
        cv2.imshow("bw_rot", bw_rot)
    #cv2.imshow("code", cv2.resize(grid, [size*10,size*10]))
    
    value = _decode(grid, V)
    
    if value is None:
        value = _decode(1-grid, V)
    
    return value

def _decode(grid, V, flipped=None, f2=False):
    if not flipped is None:
        grid = flipped
    
    lvl, mask_i = get_fmt(grid, f2)
    
    mask = MASKS[mask_i]
    unmasked = grid.copy()
    mask_area = np.ones(grid.shape)

    mask_area[:9, :9] = 0
    mask_area[:9, -8:] = 0
    mask_area[-8:, :9] = 0

    #Add alignment patterns
    locations = ALIGNMENT_PATTERN_LOCATIONS[V-1]

    if V > 1:
        for y in locations:
            for x in locations:
                #Check if not overlapping with finders
                if np.all(mask_area[y-2:y+3, x-2:x+3] == 1):
                    mask_area[y-2:y+3, x-2:x+3] = 0
                    mask_area[y-1:y+2, x-1:x+2] = 0
                    mask_area[y, x] = 0

    #Add timing patterns
    timing_length = grid.shape[0]-2*8
    mask_area[6, 8:-8] = np.zeros([timing_length])
    mask_area[8:-8, 6] = np.zeros([timing_length])

    if V >= 7:
        mask_area[-11:-8, :6] = 0
        mask_area[:6, -11:-8] = 0
    
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if mask_area[y,x] == 1 and mask(x,y):
                unmasked[y,x] = 1-unmasked[y,x]
    
    if DB_WIN: 
        cv2.namedWindow("grid", cv2.WINDOW_NORMAL)
        cv2.namedWindow("unmasked", cv2.WINDOW_NORMAL)
        #cv2.namedWindow("mask_area", cv2.WINDOW_NORMAL)
        cv2.imshow("grid", 1-grid)
        cv2.imshow("unmasked", 1-unmasked)
        #cv2.imshow("mask_area", mask_area)
    
    #Un-place data
    dir_ = -1 #-1 = up | 1 = down
    x, y = grid.shape[1]-1, grid.shape[0]-1
    i = 0
    zigzag = 0
    
    final_data_bits = ""

    while x >= 0:
        if mask_area[y,x] == 1:
            final_data_bits += str(int(unmasked[y, x]))

        if ((dir_+1)/2 + zigzag)%2 == 0:
            x -= 1

        else:
            y += dir_
            x += 1

        if y == -1 or y == grid.shape[0]:
            dir_ = -dir_
            y += dir_
            x -= 2

        else:
            zigzag = 1-zigzag

        #Vertical timing pattern
        if x == 6:
            x -= 1
    
    #Remove remainder bits
    if 2 <= V < 7:
        final_data_bits = final_data_bits[:-7]

    elif 14 <= V < 21 or 28 <= V < 35:
        final_data_bits = final_data_bits[:-3]

    elif 21 <= V < 28:
        final_data_bits = final_data_bits[:-4]
    
    ec = ERROR_CORRECTION[V-1, lvl]
    #print(ec)
    
    
    codewords = [final_data_bits[i:i+8] for i in range(0,len(final_data_bits),8)]
    
    #Only one block
    if ec[2]+ec[4] == 1:
        data_codewords = codewords[:ec[3]]
        ec_codewords = codewords[ec[3]:]

    else:
        group1, group2 = [], []
        group_ec = []
        
        for b in range(ec[2]):
            block = []
            for i in range(ec[3]):
                block.append(codewords[b+i*(ec[2]+ec[4])])
            
            group1.append(block)
        
        if ec[4] > 0:
            for b in range(ec[2], ec[2]+ec[4]):
                block = []
                for i in range(ec[5]):
                    off = 0
                    if i >= ec[3]:
                        off = (i-ec[3]+1)*ec[2]
                    block.append(codewords[b+i*(ec[2]+ec[4])-off])
                
                group2.append(block)
        
        codewords = codewords[ec[2]*ec[3]+ec[4]*ec[5]:]
        
        for b in range(ec[2]+ec[4]):
            block = []
            for i in range(ec[1]):
                block.append(codewords[b+i*(ec[2]+ec[4])])
            
            group_ec.append(block)
        
        data_codewords = sum(group1, []) + sum(group2, [])
        ec_codewords = sum(group_ec, [])
    
    #print(data_codewords)
    #print(ec_codewords)
    
    try:
        decoded = correct(data_codewords, ec_codewords)
    
    except ReedSolomonException as e:
        #print(e)
        
        if not f2:
            return _decode(grid, V, flipped, True)
        
        elif flipped is None:
            f = grid.copy()
            f = np.rot90(np.fliplr(f))
            return _decode(grid, V, f, False)
        
        else:
            #raise ReedSolomonException("Cannot decode")
            return None
    
    decoded = "".join(list(map(lambda c: f"{c.val:08b}", decoded.coefs)))
    
    mode, decoded = decoded[:4], decoded[4:]
    MODES = ["0001", "0010", "0100", "1000"]
    mode = MODES.index(mode)
    
    if 1 <= V < 10:
        char_count_len = [10,9,8,8][mode]
    elif 10 <= V < 27:
        char_count_len = [12,11,16,10][mode]
    elif 27 <= V < 41:
        char_count_len = [14,13,16,12][mode]
    
    length, decoded = decoded[:char_count_len], decoded[char_count_len:]
    length = int(length, 2)
    
    value = None
    
    if mode == 0:
        value = ""
        
        _ = length//3
        l = _ * 10
        if 10 - _ == 2:
            l += 7
        else:
            l += 4
        
        data = decoded[:l]
        groups = [data[i:i+10] for i in range(0, l, 10)]
        
        for group in groups:
            value += str(int(group,2))
        
        value = int(value)
    
    elif mode == 1:
        value = ""
        
        data = decoded[:length//2 * 11 + (length%2)*6]
        
        for i in range(0, len(data), 11):
            s = data[i:i+11]
            val = int(s, 2)
            
            if len(s) == 6:
                value += ALPHANUM[val]
            
            else:
                value += ALPHANUM[val//45]
                value += ALPHANUM[val%45]
    
    elif mode == 2:
        data = decoded[:length*8]
        data = [data[i:i+8] for i in range(0, length*8, 8)]
        data = list(map(lambda b: int(b, 2), data))
        value = bytes(data).decode("ISO-8859-1")
    
    elif mode == 3:
        value = []
        data = decoded[:length*13]
        
        for i in range(0, len(data), 13):
            val = int(data[i:i+13], 2)
            msb = val // 0xc0
            lsb = val % 0xc0
            
            dbyte = (msb << 8) + lsb
            
            if 0 <= dbyte <= 0x9ffc - 0x8140:
                dbyte += 0x8140
            
            elif 0xe040 - 0xc140 <= dbyte <= 0xebbf - 0xc140:
                dbyte += 0xc140
            
            value.append(dbyte >> 8)
            value.append(dbyte & 0xff)
        
        value = bytes(value).decode("shift_jis")
    
    #print("value:", value)
    
    return value
    
    # If unreadable
    # -> _decode(f2=True)
    # -> mirror image

class ReedSolomonException(Exception):
    pass

def correct(data, ec):
    n = len(ec)
    
    data = Poly([GF(int(cw, 2)) for cw in data+ec])
    ##print("data", list(map(lambda c:c.val, data.coefs)))
    
    syndrome = [0]*n
    corrupted = False
    for i in range(n):
        syndrome[i] = data.eval(GF.EXP[i])
        if syndrome[i].val != 0:
            corrupted = True
    
    if not corrupted:
        print("No errors")
        return data
    
    syndrome = Poly(syndrome[::-1])
    #print("syndrome", syndrome)
    
    #Find locator poly
    sigma, omega = euclidean_algorithm(Poly([GF(1)]+[GF(0) for i in range(n)]), syndrome, n)
    #print("sigma", sigma)
    #print("omega", omega)
    error_loc = find_error_loc(sigma)
    
    error_mag = find_error_mag(omega, error_loc)
    
    for i in range(len(error_loc)):
        pos = GF(error_loc[i]).log()
        pos = data.deg - pos.val - 1
        if pos < 0:
            raise ReedSolomonException("Bad error location")
        
        data.coefs[pos] += GF(error_mag[i])
    
    return data

def euclidean_algorithm(a, b, R):
    if a.deg < b.deg:
        a, b = b, a
    
    r_last = a
    r = b
    t_last = Poly([GF(0)])
    t = Poly([GF(1)])
    
    while r.deg-1 >= int(R/2):
        r_last_last = r_last
        t_last_last = t_last
        r_last = r
        t_last = t
        if r_last.coefs[0] == 0:
            raise ReedSolomonException("r_{i-1} was zero")
        
        r = r_last_last
        q = Poly([GF(0)])
        denom_lead_term = r_last.coefs[0]
        dlt_inv = denom_lead_term ** GF(-1)
        I = 0
        while r.deg >= r_last.deg and r.coefs[0] != 0:
            I += 1
            deg_diff = r.deg - r_last.deg
            scale = r.coefs[0] * dlt_inv
            q += Poly([scale]+[GF(0) for i in range(deg_diff)])
            r += r_last * Poly([scale]+[GF(0) for i in range(deg_diff)])
            q.del_lead_zeros()
            r.del_lead_zeros()
            
            if I > 100:
                raise ReedSolomonException("Too long")
        
        t = (q * t_last).del_lead_zeros() + t_last_last
        t.del_lead_zeros()
        if r.deg >= r_last.deg:
            raise ReedSolomonException("Division algorithm failed to reduce polynomial")
    
    sigma_tilde_at_zero = t.coefs[-1]
    if sigma_tilde_at_zero.val == 0:
        raise ReedSolomonException("sigma_tilde(0) was zero")
    
    inv = Poly([sigma_tilde_at_zero ** GF(-1)])
    sigma = t * inv
    omega = r * inv
    
    return [sigma, omega]

def find_error_loc(error_loc):
    num_errors = error_loc.deg-1
    if num_errors == 1:
        return [error_loc.coefs[-2].val]
    
    result = [0]*num_errors
    e = 0
    i = 1
    while i < 256 and e < num_errors:
        if error_loc.eval(GF(i)).val == 0:
            result[e] = (GF(i) ** GF(-1)).val
            e += 1
        
        i += 1
    
    if e != num_errors:
        raise ReedSolomonException("Error locator degree does not match number of roots")
    
    return result

def find_error_mag(error_eval, error_loc):
    s = len(error_loc)
    result = [0]*s
    for i in range(s):
        xi_inv = GF(error_loc[i]) ** GF(-1)
        denom = GF(1)
        for j in range(s):
            if i != j:
                denom *= GF(1) + GF(error_loc[j]) * xi_inv
        
        result[i] = ( error_eval.eval(xi_inv) * (denom ** GF(-1)) ).val
    
    return result

def get_fmt(grid ,f2=False):
    fmt1 = list(grid[0:6, 8]) + [grid[7,8], grid[8,8], grid[8,7]] + list(grid[8, 0:6][::-1])
    fmt2 = list(grid[8, -8:][::-1]) + list(grid[-7:, 8])
    
    fmt1 = "".join([str(int(b)) for b in fmt1])[::-1]
    fmt2 = "".join([str(int(b)) for b in fmt2])[::-1]
    
    if f2:
        return decode_fmt(fmt2)
    
    return decode_fmt(fmt1)

def decode_fmt(fmt):
    format_data = int(fmt, 2)
    format_data ^= 0b101010000010010
    format_data = f"{format_data:015b}"
    
    closest = None
    
    with open("./valid_format_str.txt", "r") as f:
        for i, format_str in enumerate(f):
            diff = sum(1 for a, b in zip(format_data, format_str) if a != b)
            if closest is None or diff < closest[1]:
                closest = (i, diff)
    
    lvl = closest[0] >> 3
    lvl = (5-lvl)%4
    mask = closest[0]&0b111
    return [lvl, mask]

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    
    while True:
        ret_val, img = cam.read()
        if not ret_val:
            continue
        
        cv2.imshow("src", img)
        
        try:
            value = decode(img)
            if not value is None:
                print(value)
            
        except Exception as e:
            #pass
            #raise
            print(e)
        
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()