#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generates latex tables for hamming codes

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

start = r"""  \begin{tabu}{|[2pt]c|c|c|c|c|c|c|c|[2pt]}
    \tabucline[2pt]{-}
     & 1 & 2 & 3 & 4 & 5 & 6 & 7 \\
    \tabucline[1pt]{-}
"""

end = r"""    \tabucline[2pt]{-}
  \end{tabu}
"""

class HammingError(Exception):
    pass

def encode(data, blocksize=7):
    A = start
    B = start

    result = []
    datasize = blocksize-blocksize.bit_length()
    data = list(map(int, data))
    if len(data) % datasize:
        raise HammingError(f"Length of data is not a multiple of {datasize}")

    nblocks = int(len(data)/datasize)

    last = 0
    for b in range(nblocks):
        if b > 0:
            A += "    \\hline\n"
            B += "    \\hline\n"
        A += f"    Group {b+1}"
        B += f"    Group {b+1}"
        count = 0
        for i in range(blocksize):
            A += " & "
            count += data[0]
            # Power of 2
            if (i+1)&i == 0 or i == 0:
                A += "\_"
                result.append(0)

            else:
                A += str(data[0])
                result.append(data.pop(0))
        A += " \\\\\n"

        for i in range(blocksize.bit_length()):
            p = 1 << i
            c = sum([result[b*blocksize+j] for j in range(blocksize) if (j+1)&p])
            if c%2:
                result[b*blocksize+p-1] = 1

        for i in range(blocksize):
            B += " & "
            B += str(result[b*blocksize+i])
        B += " \\\\\n"

        if count == 0:
            if last >= 2:
                A = A.rsplit("\n",2)[0]
                A += "\n    ... & ... & ... & ... & ... & ... & ... & ... \\\\\n"
                B = B.rsplit("\n",2)[0]
                B += "\n    ... & ... & ... & ... & ... & ... & ... & ... \\\\\n"
                break
            last += 1
        else:
            last = 0

    #return "".join(list(map(str, result)))
    A += end
    B += end

    return A, B

if __name__ == "__main__":
    data = "00000000001111101011000010100110101100111"
    ds = 4
    total_bits = (3**2 * 24 - 6)
    data_bits = total_bits * ds // 7
    data += "0"*(ds-len(data)%ds)
    s = ""
    i = 0
    left = data_bits-len(data)
    while len(s) < left:
        s += f"{i:0b}"
        i += 1
    s = s[:left]
    data += s
    a, b = encode(data)
    print(a)
    print(b)
