import numpy as np


########################################
"""
takes tensor products and contractions with tensors induced by SU(3) adjoint representation
"""

import itertools

def parity(new,old,sym=False): # returns the parity, 1 if even, -1 if odd
    if sym == False:
        for i in np.arange(len(new)):
            if new[0] == old[0] and new[1] == old[1] and new[2] == old[2]:
                return 1
            old = old[1:] + old[:1]
        return -1
    else:
        return 1
def gen_perm_objects(perm_list,original,sym=False):
    perm_objects = []
    for perm_el in perm_list:
        perm_objects.append({'perm':perm_el,'parity':parity(perm_el,original,sym)})
    return perm_objects

listA = [0, 1, 2]
perm = itertools.permutations(listA)

obj_list = gen_perm_objects(perm,listA)
print("writing out the permutations and parities")
for obj in obj_list:
    print(obj)

M = np.zeros((3,3,3))
def make_eps(obj_list):
    M = np.zeros((3,3,3))
    for obj in obj_list:
        perm = obj['perm']
        i,j,k = perm[0],perm[1],perm[2]
        M[i][j][k] =1* obj['parity']
    return M

M = make_eps(obj_list)
################################# summation
import sympy as sp

#from sympy.abc import x,y,z
import tensorflow as tf


v1=np.array([sp.Symbol('x_1'),sp.Symbol('x_2'),sp.Symbol('x_3')])
v2= np.array([sp.Symbol('y_1'),sp.Symbol('y_2'),sp.Symbol('y_3')])

"""
A =np.array([[x,y],[z,d]])
B =np.array([[m,n],[g,h]])
print(A)
print(B)
print(np.tensordot(A,B,axes=((1),(0))))

M=np.tensordot(M,v1,axes=((0),(0)))
M=np.tensordot(M,v2,axes=((0),(0)))
for el in M:
    print(el)
"""
def gen_f():
    a = []
    half = sp.sympify(1)/sp.sympify(2)
    a.append({'ind':[0,1,2],'num':sp.sympify(1)})
    a.append({'ind':[3,4,7],'num':sp.sqrt(3)/2})
    a.append({'ind':[5,6,7],'num':sp.sqrt(3)/2})
    a.append({'ind':[0,3,6],'num':half})
    a.append({'ind':[1,3,5],'num':half})
    a.append({'ind':[1,4,6],'num':half})
    a.append({'ind':[2,3,4],'num':half})
    a.append({'ind':[4,0,5],'num':half})
    a.append({'ind':[5,2,6],'num':half})
    return a
def gen_d():
    a = []
    half = sp.sympify(1)/sp.sympify(2)
    inv_sq_r_3 = sp.sympify(1)/sp.sqrt(sp.sympify(3))
    a.append({'ind':[0,0,7],'num':inv_sq_r_3})
    a.append({'ind':[1,1,7],'num':inv_sq_r_3})
    a.append({'ind':[2,2,7],'num':inv_sq_r_3})
    a.append({'ind':[7,7,7],'num':-inv_sq_r_3})
    a.append({'ind':[0,3,5],'num':half})
    a.append({'ind':[0,4,6],'num':half})
    a.append({'ind':[1,3,6],'num':-half})
    a.append({'ind':[1,4,5],'num':half})
    a.append({'ind':[2,3,3],'num':half})
    a.append({'ind':[2,4,4],'num':half})
    a.append({'ind':[2,5,5],'num':-half})
    a.append({'ind':[2,6,6],'num':-half})
    a.append({'ind':[3,3,7],'num':-half*inv_sq_r_3})
    a.append({'ind':[4,4,7],'num':-half*inv_sq_r_3})
    a.append({'ind':[5,5,7],'num':-half*inv_sq_r_3})
    a.append({'ind':[6,6,7],'num':-half*inv_sq_r_3})
    return a

def fill_up_once(obj_list,M,numm):
    for obj in obj_list:
        perm = obj['perm']
        i,j,k = perm[0],perm[1],perm[2]
        M[i][j][k] =sp.sympify(numm * sp.sympify(obj['parity'],evaluate=False),evaluate=False)
    return M
def gen_tensor(component_list,sym=False):
    f_list = component_list
    M = np.full((8,8,8),sp.sympify(0))
    for f in f_list:
        perm = itertools.permutations(f['ind'])
        obj_list = gen_perm_objects(perm,f['ind'],sym)
        M=fill_up_once(obj_list,M,f['num'])
    return M
F = gen_tensor(gen_f())
D = gen_tensor(gen_d(),True)

n_1=sp.Symbol('n_1')
n_2=sp.Symbol('n_2')
n_3=sp.Symbol('n_3')
n_4=sp.Symbol('n_4')
n_5=sp.Symbol('n_5')
n_6=sp.Symbol('n_6')
n_7=sp.Symbol('n_7')
n_8=sp.Symbol('n_8')#sp.sqrt(1-(n_1*n_1-n_2*n_2-n_2*n_2-n_3*n_3-n_4*n_4-n_5*n_5-n_6*n_6-n_7*n_7))#
n = np.array([n_1,n_2,n_3,n_4,n_5,n_6,n_7,n_8])
m = np.array([sp.Symbol('m_1'),sp.Symbol('m_2'),sp.Symbol('m_3'),sp.Symbol('m_4'),sp.Symbol('m_5'),sp.Symbol('m_6'),sp.Symbol('m_7'),sp.Symbol('m_8')])


M1=np.tensordot(F,n,axes=((0),(0)))
M1=np.tensordot(M1,m,axes=((0),(0)))
print("n wedge m")
#for el in M1:
#    print(sp.simplify(el))

print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
print("n star m")
M2=np.tensordot(D,n,axes=((0),(0)))
M2=np.tensordot(M2,m,axes=((0),(0)))
"""
for el in M2:
    print(sp.simplify(el))
"""
print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
print("n star n")
n_star_n=np.tensordot(D,n,axes=((0),(0)))
n_star_n=np.tensordot(n_star_n,n,axes=((0),(0)))
#for el in n_star_n:
#    print(sp.simplify(el))

print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
print("n star (n wedge m)")
M3=np.tensordot(D,n,axes=((0),(0)))
M3=np.tensordot(M3,M1,axes=((0),(0)))
#for el in M3:
#    print(sp.simplify(el))

#print(sp.simplify(np.tensordot(D,F,1)))
print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
print("n wedge (n star m)")
M4=np.tensordot(F,n,axes=((0),(0)))
M4=np.tensordot(M4,M2,axes=((0),(0)))
#for el in M3:
#    print(sp.simplify(el))

print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
print("m wedge (n star n)")
M5=np.tensordot(F,m,axes=((0),(0)))
M5=np.tensordot(M5,n_star_n,axes=((0),(0)))

"""
print("yeahhh")
for i in np.arange(len(M4)):
    print(sp.simplify(M4[i]-M3[i]))
print("last_not")
#print(sp.simplify(np.tensordot(M5,n,1)))
"""

print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
print("n star[m wedge (n star n)]")
M6=np.tensordot(D,n,axes=((0),(0)))
M6=np.tensordot(M6,M5,axes=((0),(0)))
print("last")
#for el in M6:
#    print(sp.simplify(el))

print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
print("n star (n star m)")
M7=np.tensordot(D,n,axes=((0),(0)))
M7=np.tensordot(M7,M2,axes=((0),(0)))
#for el in M7:
#    print(sp.simplify(el))

print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
print("n wedge (n wedge m)")
M8=np.tensordot(F,n,axes=((0),(0)))
M8=np.tensordot(M8,M1,axes=((0),(0)))
#for el in M8:
#    print(sp.simplify(el))
#print(sp.simplify(np.tensordot(D,F,1)))
print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
print("m star (n star n)")
M9=np.tensordot(D,m,axes=((0),(0)))
M9=np.tensordot(M9,n_star_n,axes=((0),(0)))

"""
a = np.tensordot(n_star_n,m,1)
b = (sp.sqrt(3))*np.tensordot(M9,n_star_n,1)
print(sp.simplify(a))
"""
"""
for i in np.arange(len(M9)):
    print(sp.simplify(M7[i]))
    print("break")
    print(sp.simplify(M9[i]))
    print("break")
    print("break")
    print("break")
"""
print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
print("n wedge [n wedge (n wedge m)]")
M10=np.tensordot(F,n,axes=((0),(0)))
M10=np.tensordot(M10,M8,axes=((0),(0)))

print("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH")
print("bravado")
M11=np.tensordot(n_star_n,n_star_n,1)
print(sp.simplify(M11))
