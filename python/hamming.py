#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This module provides encoding and decoding functions for Hamming codes

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

class HammingError(Exception):
    pass

def encode(data, blocksize=7):
    result = []
    datasize = blocksize-blocksize.bit_length()
    data = list(map(int, data))
    if len(data) % datasize:
        raise HammingError(f"Length of data is not a multiple of {datasize}")

    nblocks = int(len(data)/datasize)

    for b in range(nblocks):
        for i in range(blocksize):
            # Power of 2
            if (i+1)&i == 0 or i == 0:
                result.append(0)

            else:
                result.append(data.pop(0))

        for i in range(blocksize.bit_length()):
            p = 1 << i
            c = sum([result[b*blocksize+j] for j in range(blocksize) if (j+1)&p])
            if c%2:
                result[b*blocksize+p-1] = 1

    return "".join(list(map(str, result)))

def decode(data, blocksize=7):
    result = []
    datasize = blocksize-blocksize.bit_length()
    data = list(map(int, data))
    if len(data) % blocksize:
        raise HammingError(f"Length of data is not a multiple of {blocksize}")

    nblocks = int(len(data)/blocksize)
    errors = 0

    for b in range(nblocks):
        pos = 0
        for i in range(blocksize.bit_length()):
            p = 1 << i
            c = sum([data[b*blocksize+j] for j in range(blocksize) if (j+1)&p])
            if c%2:
                pos |= p

        if pos != 0:
            if pos > blocksize:
                raise HammingError("Too many errors")
                return

            errors += 1
            data[b*blocksize+pos-1] = 1-data[b*blocksize+pos-1]

        for i in range(1, blocksize):
            if (i+1)&i != 0:
                result.append(data[b*blocksize+i])
    
    return "".join(list(map(str, result))), errors

if __name__ == "__main__":
    #print("10011010")
    print(encode("10011010"))
    #print(decode("011100101010"))
    print(decode("00110011011010101101001010101011010101010111001100110011"))
    print(decode("01000011011010101100001010101011110101000111001100111011"))
