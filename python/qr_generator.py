#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pygame
import time
import sys

S_EMPTY = 1
S_SEP = 2
S_FINDER = 4
S_ALIGN = 8
S_TIMING = 16
S_RESERVED = 32
S_DATA = 64
S_BYTES = 128
S_MASK = 256

STEPS = S_MASK|S_DATA|S_BYTES|S_RESERVED|S_ALIGN|S_TIMING|S_FINDER|S_SEP|S_EMPTY
#STEPS = S_MASK|S_RESERVED|S_ALIGN|S_TIMING|S_FINDER|S_EMPTY
#STEPS = 0
#STEPS = S_MASK|S_DATA

pygame.init()
font = pygame.font.SysFont("ubuntu", 16)
win = None
coords = True  # turn on coordinates

def log(msg):
    print(f"<[---]> {msg} <[---]>")

class GF:
    """Galois field element"""
    
    def __init__(self, val):
        self.val = val

    def copy(self):
        return GF(self.val)

    # Addition
    def __add__(self, n):
        return GF(self.val ^ n.val)
    
    # Subtraction
    def __sub__(self, n):
        return GF(self.val ^ n.val)
    
    # Multiplication
    def __mul__(self, n):
        if self.val == 0 or n.val == 0:
            return GF(0)

        return GF.EXP[GF.LOG[self.val].val + GF.LOG[n.val].val].copy()
    
    # Division
    def __truediv__(self, n):
        if n.val == 0:
            raise ZeroDivisionError
        if self.val == 0:
            return GF(0)

        return GF.EXP[(GF.LOG[self.val].val + 255 - GF.LOG[n.val].val)%255].copy()

    # Power
    def __pow__(self, n):
        return GF.EXP[(GF.LOG[self.val].val * n.val)%255].copy()
    
    # Representation -> string
    def __repr__(self):
        return self.val.__repr__()

# Compute exponents and logs for all element of the Galois field
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
    """
    Polynomial
    
    Coefficients are in the order of largest to lowest degree:
    ax^2 + bx + c -> coefs = [a, b, c]
    """
    
    def __init__(self, coefs):
        self.coefs = coefs.copy()

    @property
    def deg(self):
        return len(self.coefs)

    def copy(self):
        return Poly(self.coefs)
    
    # Addition
    def __add__(self, p):
        d1, d2 = self.deg, p.deg
        deg = max(d1,d2)
        result = [GF(0) for i in range(deg)]

        for i in range(d1):
            result[i + deg - d1] = self.coefs[i]

        for i in range(d2):
            result[i + deg - d2] += p.coefs[i]

        return Poly(result)
    
    # Multiplication
    def __mul__(self, p):
        result = [GF(0) for i in range(self.deg+p.deg-1)]

        for i in range(p.deg):
            for j in range(self.deg):
                result[i+j] += self.coefs[j] * p.coefs[i]

        return Poly(result)
    
    # Division
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
    
    # Representation -> string
    def __repr__(self):
        return f"<Poly {self.coefs}>"

# Inspired by nayuki's Creating a QR Code step by step
# bibtex key: nayuki_qr_js
# https://github.com/nayuki/Nayuki-web-published-code/blob/dfb110475327271e3b7279a432e2d1a1298815ad/creating-a-qr-code-step-by-step/creating-qr-code-steps.js
class History:
    """Widths history for mask evaluation, crit. 3"""
    
    def __init__(self):
        self.widths = [0]*7
        self.widths[-1] = 4
        self.colors = [0]*4
        self.color = 0
    
    # Add module to history, returns number of patterns found
    def add(self, col):
        s = 0
        self.colors.append(col)
        if col != self.color:
            self.color = col
            s = self.check()
            self.widths.pop(0)
            self.colors = self.colors[-sum(self.widths)-1:]
            self.widths.append(0)
        
        self.widths[-1] += 1
        return s
    
    # Check for patterns in the history
    def check(self):
        n = self.widths[1]
        
        # Only black on white
        if self.colors[self.widths[0]] != 1: return 0
        
        # if 1:1:3:1:1
        if n > 0 and self.widths[2] == n and self.widths[3] == n*3 and self.widths[4] == n and self.widths[5] == n:
            # check if 4:1:1:3:1:1 + check if 1:1:3:1:1:4
            return int(self.widths[0] >= 4) + int(self.widths[6] >= 4)
        
        return 0
    
    # Final check
    def final(self):
        for i in range(4):
            self.add(0)
        
        return self.check()

class QR:
    TYPES = ["numeric", "alphanumeric", "byte", "kanji",    "?"]
    LEVELS = ["L","M","Q","H",    "?"]
    MODES = ["0001", "0010", "0100", "1000"]
    ALPHANUM = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
    VERSIONS = []
    ERROR_CORRECTION = []
    FINDER = ["1111111","1000001","1011101","1011101","1011101","1000001","1111111"]
    FINDER = np.array([list(map(int, _)) for _ in FINDER])

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

    def __init__(self, data, level=0):
        pygame.display.set_caption("QR Gen - Init")
        log(f"Content: {data}")
        log(f"EC Level: {self.LEVELS[level]}")
        self.bits = ""
        self.data = data
        self.level = level
        self.type = -1
        self.version = -1
        self.analyse_type()
        self.compute_version()
        self.build_char_count_indicator()
        self.encode()
        self.separate_codewords()
        self.create_matrix()

    def load_versions():
        with open("qr_versions.txt", "r") as f:
            versions = f.read().split("\n\n")
            for v in versions:
                lvls = [list(map(int, lvl.split("\t"))) for lvl in v.split("\n")]
                QR.VERSIONS.append(lvls)

        QR.VERSIONS = np.array(QR.VERSIONS)

    def load_ec():
        with open("error_correction.txt", "r") as f:
            ecs = f.read().split("\n\n")
            for ec in ecs:
                lvls = [list(map(int, lvl.split("\t"))) for lvl in ec.split("\n")]
                lvls = [lvl + [0]*(6-len(lvl)) for lvl in lvls]

                QR.ERROR_CORRECTION.append(lvls)

        QR.ERROR_CORRECTION = np.array(QR.ERROR_CORRECTION)

    def __repr__(self):
        return "<QR: {} - {} (V{})>".format(
            QR.LEVELS[self.level],
            QR.TYPES[self.type].title(),
            "?" if self.version == -1 else self.version+1
        )

    def analyse_type(self):
        pygame.display.set_caption("QR Gen - Type analysis")
        if self.data.isnumeric():
            self.type = 0

        elif set(self.data).issubset(set(QR.ALPHANUM)):
            self.type = 1

        else:
            try:
                self.data.encode("ISO-8859-1")
                self.type = 2

            except:
                self.type = 3

        self.bits += self.MODES[self.type]
        log(f"Type: {self.TYPES[self.type]}")

    def compute_version(self):
        pygame.display.set_caption("QR Gen - Version computation")
        self.version = min(np.where(QR.VERSIONS[:, self.level, self.type] >= len(self.data))[0])
        log(f"Version: {self.version+1}")

    def get_char_count_len(self):
        if 0 <= self.version < 9:
            return [10,9,8,8][self.type]
        elif 9 <= self.version < 26:
            return [12,11,16,10][self.type]
        elif 26 <= self.version < 40:
            return [14,13,16,12][self.type]

    def build_char_count_indicator(self):
        pygame.display.set_caption("QR Gen - Char count ind")
        length = self.get_char_count_len()
        indicator = f"{{:0{length}b}}".format(len(self.data))
        self.bits += indicator
        log(f"Char count indicator: {indicator}")

    def encode(self):
        pygame.display.set_caption("QR Gen - Encoding")
        if self.type == 0:
            groups = [self.data[i:i+3] for i in range(0,len(self.data),3)]

            for group in groups:
                group = int(group)
                sgroup = str(group)

                if len(sgroup) == 3:
                    s = "{:010b}"
                elif len(sgroup) == 2:
                    s = "{:07b}"
                else:
                    s = "{:04b}"

                self.bits += s.format(group)

        elif self.type == 1:
            data = self.data
            last = None
            if len(data)%2 == 1:
                last = data[-1]
                data = data[:-1]

            for i in range(0, len(data), 2):
                val1 = self.ALPHANUM.index(data[i])
                val2 = self.ALPHANUM.index(data[i+1])
                val = val1*45 + val2
                self.bits += f"{val:011b}"

            if not last is None:
                self.bits += "{:06b}".format(self.ALPHANUM.index(last))

        elif self.type == 2:
            data = self.data.encode("ISO-8859-1")
            self.bits += "".join(list(map("{:08b}".format, data)))

        elif self.type == 3:
            data = list(self.data.encode("shift_jis"))

            #Combine double bytes
            data = [data[i*2]<<8 | data[i*2+1] for i in range(len(data)//2)]

            for dbyte in data:
                if 0x8140 <= dbyte <= 0x9ffc:
                    dbyte = dbyte - 0x8140

                elif 0xe040 <= dbyte <= 0xebbf:
                    dbyte = dbyte - 0xc140

                msb = dbyte >> 8
                lsb = dbyte & 0xff

                val = msb * 0xc0 + lsb
                self.bits += f"{val:013b}"
        
        log(f"Encoded: {[self.bits[i:i+8] for i in range(0,len(self.bits),8)]}")

        ec = self.ERROR_CORRECTION[self.version, self.level]
        req_bits = ec[0]*8

        #Terminator
        self.bits += "0"*(min(4, req_bits-len(self.bits)))

        #Pad to multiple of 8
        if len(self.bits) % 8 != 0:
            self.bits += "0"*(8-len(self.bits)%8)

        #Pad to required bits
        if len(self.bits) < req_bits:
            for i in range((req_bits-len(self.bits))//8):
                self.bits += ["11101100","00010001"][i%2]
        
        log(f"Padded: {[self.bits[i:i+8] for i in range(0,len(self.bits),8)]}")

    def separate_codewords(self):
        pygame.display.set_caption("QR Gen - Separating codewords")
        ec = self.ERROR_CORRECTION[self.version, self.level]
        blocks = []
        ec_codewords = []

        codeword = 0
        gen_poly = self.get_generator_poly(ec[1])
        log(f"Gen poly: {gen_poly}")
        
        #print(self.bits)
        for i in range(ec[2]):
            block = []
            for j in range(ec[3]):
                block.append(self.bits[codeword*8:codeword*8+8])
                codeword += 1

            blocks.append(block)
            msg_poly = Poly(list(map(lambda b: GF(int(b,2)), block)))
            log(f"Msg poly (1-{i}): {msg_poly}")

            quotient, remainder = msg_poly / gen_poly
            log(f"EC poly (1-{i}): {remainder}")
            ec_cwds = [f"{c.val:08b}" for c in remainder.coefs]
            ec_codewords.append(ec_cwds)

        #If group 2
        if ec[4] != 0:
            for i in range(ec[4]):
                block = []
                for j in range(ec[5]):
                    block.append(self.bits[codeword*8:codeword*8+8])
                    codeword += 1

                blocks.append(block)
                msg_poly = Poly(list(map(lambda b: GF(int(b,2)), block)))
                log(f"Msg poly (2-{i}): {msg_poly}")

                quotient, remainder = msg_poly / gen_poly
                log(f"EC poly (2-{i}): {remainder}")
                ec_cwds = [f"{c.val:08b}" for c in remainder.coefs]
                ec_codewords.append(ec_cwds)
        
        self.final_data_bits = ""

        if len(blocks) == 1:
            dbits = "".join(["".join(block) for block in blocks])
            ec_bits = "".join(["".join(cwd) for cwd in ec_codewords])
            log(f"EC bits: {[ec_bits[i:i+8] for i in range(0,len(ec_bits),8)]}")
            self.final_data_bits = dbits + ec_bits

        else:
            #Interleave data codewords
            for i in range(max(ec[3], ec[5])):
                for block in blocks:
                    if i < len(block):
                        self.final_data_bits += block[i]

            #Interleave error correction codewords
            for i in range(ec[1]):
                for block in ec_codewords:
                    self.final_data_bits += block[i]

        #Add remainder bits
        if 1 <= self.version < 6:
            self.final_data_bits += "0"*7
            log(f"Add 7 remainder bits")

        elif 13 <= self.version < 20 or 27 <= self.version < 34:
            self.final_data_bits += "0"*3
            log(f"Add 3 remainder bits")

        elif 20 <= self.version < 27:
            self.final_data_bits += "0"*4
            log(f"Add 4 remainder bits")
        
        print_bytes(self.final_data_bits)

    def get_generator_poly(self, n):
        poly = Poly([GF(1)])

        for i in range(n):
            poly *= Poly([GF(1), GF(2)**GF(i)])

        return poly

    def get_alignment_pattern_locations(self):
        return QR.ALIGNMENT_PATTERN_LOCATIONS[self.version]

    def create_matrix(self):
        size = self.version*4+21
        log(f"Size: {size}")
        self.matrix = np.zeros([size, size])-1 #-1: empty | -0.5: reserved | 0: white | 1: black
        
        pygame.display.set_caption("QR Gen - Matrix")
        if STEPS & S_EMPTY: self.show(step=True)

        #Add separator
        self.matrix[0:8, 0:8] = 0
        self.matrix[-8:, 0:8] = 0
        self.matrix[0:8, -8:] = 0
        
        pygame.display.set_caption("QR Gen - Separator")
        if STEPS & S_SEP: self.show(step=True)

        #Place finders
        self.matrix[0:7, 0:7] = QR.FINDER
        self.matrix[-7:, 0:7] = QR.FINDER
        self.matrix[0:7, -7:] = QR.FINDER
        
        pygame.display.set_caption("QR Gen - Finder patterns")
        if STEPS & S_FINDER: self.show(step=True)

        #Add alignment patterns
        locations = self.get_alignment_pattern_locations()
        log(f"Alignment patterns: {locations}")

        if self.version > 0:
            for y in locations:
                for x in locations:
                    #Check if not overlapping with finders
                    if np.all(self.matrix[y-2:y+3, x-2:x+3] == -1):
                        self.matrix[y-2:y+3, x-2:x+3] = 1
                        self.matrix[y-1:y+2, x-1:x+2] = 0
                        self.matrix[y, x] = 1
        
        pygame.display.set_caption("QR Gen - Alignment patterns")
        if STEPS & S_ALIGN: self.show(step=True)

        #Add timing patterns
        timing_length = size-2*8
        self.matrix[6, 8:-8] = np.resize([1,0],timing_length)
        self.matrix[8:-8, 6] = np.resize([1,0],timing_length)
        
        pygame.display.set_caption("QR Gen - Timing patterns")
        if STEPS & S_TIMING: self.show(step=True)

        #Add reserved areas
        self.matrix[self.version*4+13,8] = 1 #Black module
        self.matrix[:9, :9] = np.maximum(self.matrix[:9, :9], -0.5) #Top-left
        self.matrix[-8:, 8] = np.maximum(self.matrix[-8:, 8], -0.5) #Bottom-left
        self.matrix[8, -8:] = np.maximum(self.matrix[8, -8:], -0.5) #Top-right

        if self.version >= 6:
            self.matrix[-11:-8, :6] = -0.5
            self.matrix[:6, -11:-8] = -0.5
        
        pygame.display.set_caption("QR Gen - Reserved areas")
        if STEPS & S_RESERVED: self.show(step=True)

        #Place data
        dir_ = -1 #-1 = up | 1 = down
        x, y = size-1, size-1
        i = 0
        zigzag = 0

        mask_area = self.matrix == -1
        pygame.display.set_caption("QR Gen - Data layout")
        print(self.matrix.tolist())

        while x >= 0:
            if self.matrix[y,x] == -1:
                self.matrix[y,x] = self.final_data_bits[i]

                i += 1

                if STEPS & S_DATA:
                    if not (STEPS & S_BYTES) or i%8==0:
                        self.show()
                        time.sleep(0.01)

            if ((dir_+1)/2 + zigzag)%2 == 0:
                x -= 1

            else:
                y += dir_
                x += 1

            if y == -1 or y == size:
                dir_ = -dir_
                y += dir_
                x -= 2

            else:
                zigzag = 1-zigzag

            #Vertical timing pattern
            if x == 6:
                x -= 1

        if STEPS & S_DATA: self.show(step=True)

        score, mask, matrix = self.try_masks(mask_area)

        self.matrix = np.where(mask_area, matrix, self.matrix)
        
        pygame.display.set_caption("QR Gen - Mask")
        if STEPS & S_MASK: self.show(step=True)

        #Format string
        format_str = f"{(5-self.level)%4:02b}{mask:03b}"
        format_str += "0"*10
        format_str.lstrip("0")
        log(f"Format str: {format_str}")

        gen_poly = 0b10100110111
        format_poly = int(format_str,2)

        while format_poly.bit_length() > 10:
            g = gen_poly << (format_poly.bit_length()-gen_poly.bit_length())
            format_poly ^= g

        log(f"Remainder: {format_poly:b}")
        format_data = int(format_str,2) + format_poly
        format_data ^= 0b101010000010010
        format_data = f"{format_data:015b}"
        log(f"XORed: {format_data}")

        for i in range(15):
            y1, x1 = min(8,15-i), min(7,i)
            if i >= 6:
                x1 += 1
                if i >= 9:
                    y1 -= 1
            y2, x2 = self.matrix.shape[0]-i-1 if i < 7 else 8, 8 if i < 7 else self.matrix.shape[1]+i-15

            self.matrix[y1, x1] = format_data[i]
            self.matrix[y2, x2] = format_data[i]

        #Version information
        if self.version >= 6:
            gen_poly = 0b1111100100101
            version_info_poly = int(self.version+1)<<12

            while version_info_poly.bit_length() > 12:
                g = gen_poly << (version_info_poly.bit_length()-gen_poly.bit_length())
                version_info_poly ^= g

            version_info_data = ((self.version+1)<<12) + version_info_poly
            version_info_data = f"{version_info_data:018b}"

            ox1, oy1 = 5, self.matrix.shape[0]-9
            ox2, oy2 = self.matrix.shape[1]-9, 5
            for i in range(18):
                self.matrix[oy1 - i%3, ox1 - i//3] = version_info_data[i]
                self.matrix[oy2 - i//3, ox2 - i%3] = version_info_data[i]

    def try_masks(self, mask_area):
        best = [None,None,None] #score, i, matrix

        for i in range(8):
            mask = QR.MASKS[i]
            mat = self.matrix.copy()

            for y in range(self.matrix.shape[0]):
                for x in range(self.matrix.shape[1]):
                    if mask_area[y,x] and mask(x,y):
                        mat[y,x] = 1-mat[y,x]
            
            
            #Format string
            format_str = f"{(5-self.level)%4:02b}{i:03b}"
            format_str += "0"*10
            format_str.lstrip("0")

            gen_poly = 0b10100110111
            format_poly = int(format_str,2)

            while format_poly.bit_length() > 10:
                g = gen_poly << (format_poly.bit_length()-gen_poly.bit_length())
                format_poly ^= g

            format_data = int(format_str,2) + format_poly
            format_data ^= 0b101010000010010
            format_data = f"{format_data:015b}"

            for j in range(15):
                y1, x1 = min(8,15-j), min(7,j)
                if j >= 6:
                    x1 += 1
                    if j >= 9:
                        y1 -= 1
                y2, x2 = mat.shape[0]-j-1 if j < 7 else 8, 8 if j < 7 else mat.shape[1]+j-15

                mat[y1, x1] = format_data[j]
                mat[y2, x2] = format_data[j]
            
            score = self.evaluate(mat.copy(), i)
            
            if best[0] is None or score < best[0]:
                best = [score, i, mat]
        
        return best

    def evaluate(self, matrix, i):
        score = 0

        matrix = np.where(matrix < 0, 0, matrix)
        
        s1, s2, s3, s4 = 0, 0, 0, 0
        
        #Condition 1 (horizontal)
        for y in range(matrix.shape[0]):
            col, count = -1, 0
            for x in range(matrix.shape[1]):
                if matrix[y,x] != col:
                    count = 0
                    col = matrix[y,x]
                count += 1

                if count == 5:
                    score += 3
                    s1 += 3
                elif count > 5:
                    score += 1
                    s1 += 1

        #Condition 1 (vertical)
        for x in range(matrix.shape[1]):
            col, count = -1, 0
            for y in range(matrix.shape[0]):
                if matrix[y,x] != col:
                    count = 0
                    col = matrix[y,x]
                count += 1

                if count == 5:
                    score += 3
                    s1 += 3
                elif count > 5:
                    score += 1
                    s1 += 1

        #Condition 2
        for y in range(matrix.shape[0]-1):
            for x in range(matrix.shape[1]-1):
                zone = matrix[y:y+2, x:x+2]
                if np.all(zone == zone[0,0]):
                    score += 3
                    s2 += 3
        
        #Condition 3 (horizontal)
        for y in range(matrix.shape[0]):
            hist = History()
            for x in range(matrix.shape[1]):
                s = hist.add(matrix[y,x])
                score += s*40
                s3 += s*40
            
            s = hist.final()
            score += s*40
            s3 += s*40
        
        #Condition 3 (vertical)
        for x in range(matrix.shape[1]):
            hist = History()
            for y in range(matrix.shape[0]):
                s = hist.add(matrix[y,x])
                score += s*40
                s3 += s*40
            
            s = hist.final()
            score += s*40
            s3 += s*40

        #Condition 4
        total = matrix.shape[0]*matrix.shape[1]
        dark = np.sum(matrix == 1)
        percent = 100*dark//total
        p1 = percent-(percent%5)
        p2 = p1+5
        p1, p2 = abs(p1-50)/5, abs(p2-50)/5
        score += min(p1,p2)*10
        s4 += min(p1,p2)*10
        
        log(f"mask {i}: {s1} + {s2} + {s3} + {s4} = {score}")

        return score

    def show(self, pos=None, step=False):
        global win
        
        events = pygame.event.get()
        
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    pygame.image.save(win, "/tmp/qr.jpg")
        
        m = ((self.matrix.copy()+2)%3)*127
        mat = np.ones((m.shape[0]+8, m.shape[1]+8))*255
        mat[4:-4, 4:-4] = m

        if not pos is None:
            mat[pos[1]+4, pos[0]+4] = 50
        
        size = 15
        if win is None:
            win = pygame.display.set_mode([mat.shape[0]*size, mat.shape[0]*size])
        
        win.fill((255,255,255))
        
        for y in range(mat.shape[0]):
            for x in range(mat.shape[1]):
                col = mat[y, x]
                col = (col, col, col)
                pygame.draw.rect(win, col, [x*size, y*size, size, size])
        
        if coords:
            N = 6
            space = (mat.shape[0]-8)/(N-1)
            margin = 4*size
            SIZE = (mat.shape[0]-8)*size
            pygame.draw.lines(win, (0,0,0), True, [
                (margin, margin),(margin+SIZE, margin),
                (margin+SIZE, margin+SIZE),(margin, margin+SIZE)
            ])
            for i in range(N):
                n = int(round(space*i))
                d = size * n
                pygame.draw.line(win, (0,0,0), [margin+d, margin], [margin+d, margin-15])
                pygame.draw.line(win, (0,0,0), [margin, margin+d], [margin-15, margin+d])
                pygame.draw.line(win, (0,0,0), [margin+d, margin+SIZE], [margin+d, margin+SIZE+15])
                pygame.draw.line(win, (0,0,0), [margin+SIZE, margin+d], [margin+SIZE+15, margin+d])
                text = font.render(str(n), True, (0,0,0))
                win.blit(text, [margin+d-text.get_width()/2, margin-30-text.get_height()/2])
                win.blit(text, [margin-30-text.get_width()/2, margin+d-text.get_height()/2])
                win.blit(text, [margin+d-text.get_width()/2, margin+SIZE+30-text.get_height()/2])
                win.blit(text, [margin+SIZE+30-text.get_width()/2, margin+d-text.get_height()/2])
        
        pygame.display.flip()

        if step:
            input("Press Enter to continue")

def print_bytes(bytes_, int_=False):
    result = ""
    for i in range(len(bytes_)//8):
        if int_:
            result += str(int(bytes_[i*8:i*8+8],2)) + " "
        else:
            result += bytes_[i*8:i*8+8] + " "

    result += bytes_[-(len(bytes_)%8):]

    print(result.strip())

QR.load_versions()
QR.load_ec()

if __name__ == "__main__":
    np.set_printoptions(linewidth=200)
    pygame.display.set_caption("QR Gen")
    #qr = QR("8675309", 0)
    #qr = QR("HELLO WORLD", 2)
    qr = QR("Hello, World!", 1)
    #qr = QR("茗荷", 2)
    #qr = QR("Hello, world! How are you doing ? I'm doing great, thank you ! Today is quite a sunny day, isn't it ?", 3)
    #qr = QR("https://aufildeverre.ch/", 3)
    #qr = QR("QR Code Symbol", 1)
    #qr = QR("Attention !", 3)
    #qr = QR("Lycacode", 0)
    
    print(qr)
    
    pygame.display.set_caption("QR Gen - Final")
    qr.show(step=False)
    input("Press Enter to quit")
    events = pygame.event.get()
    
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:
                pygame.image.save(win, "/tmp/qr.jpg")