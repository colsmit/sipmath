import numpy as np
from scipy.optimize import linprog
import cvxopt

A_ub = np.loadtxt('A_ub.txt', delimiter=',')
A_eq = np.loadtxt('A_eq.txt', delimiter=',')
b_ub = np.loadtxt('b_ub.txt', delimiter=',')
b_eq = np.loadtxt('b_eq.txt', delimiter=',')
c = np.loadtxt('c.txt', delimiter=',')

#lp_sol = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='simplex', options={"maxiter": 5000, "tol": 1.0e-10, "disp": False})
c = cvxopt.matrix(c)
G = cvxopt.matrix(A_ub)
h = cvxopt.matrix(b_ub)
A = cvxopt.matrix(A_eq)
b = cvxopt.matrix(b_eq)

lp_sol = cvxopt.solvers.conelp(c=c, G=G, h=h, A=A, b=b, solver='glpk')

print(lp_sol)