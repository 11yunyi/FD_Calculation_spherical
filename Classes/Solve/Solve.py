import scipy.sparse.linalg as la
#import scipy
#import numpy as np


# provide class to Solve an system of equations
class Solve:
    def __init__(self):
        pass

    def solve(self, A, b):
        print ("Solve!")
        #return scipy.sparse.linalg.solve(A, b)
        return la.spsolve(A, b)

