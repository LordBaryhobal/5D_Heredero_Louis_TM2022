#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prints steps of Reed-Solomon decoding

(C) 2022 Louis Heredero  louis.heredero@edu.vs.ch
"""

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
        #return f"GF({self.val})"
    
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
            print("sub:", p*Poly([coef]))

            for j in range(p.deg):
                dividend[i+j] -= p.coefs[j] * coef
            
            print("rem:", dividend)
            print()

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

def get_generator_poly(n):
    poly = Poly([GF(1)])

    for i in range(n):
        poly *= Poly([GF(1), GF(2)**GF(i)])

    return poly

class ReedSolomonException(Exception):
    pass

def correct(data, ec):
    n = len(ec)
    
    data = Poly([GF(int(cw, 2)) for cw in data+ec])
    ##print("data", list(map(lambda c:c.val, data.coefs)))
    print("r(x)", data)
    
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
    print("syndrome", syndrome)
    
    #Find locator poly
    sigma, omega = euclidean_algorithm(Poly([GF(1)]+[GF(0) for i in range(n)]), syndrome, n)
    print("locator", sigma)
    print("evaluator", omega)
    error_loc = find_error_loc(sigma)
    print("location", error_loc)
    
    error_mag = find_error_mag(omega, error_loc)
    print("mag", error_mag)
    
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

if __name__ == "__main__":
    m = Poly([GF(67),GF(111),GF(100),GF(101),GF(115)])
    g = get_generator_poly(4)
    print()
    print(g)
    print(m/g)
    print()
    
    data = [67,111,110,101,115]
    #ec = [119,123,82]
    ec = [50,166,245,58]
    
    data = [f"{n:08b}" for n in data]
    ec = [f"{n:08b}" for n in ec]
    
    print(correct(data, ec))