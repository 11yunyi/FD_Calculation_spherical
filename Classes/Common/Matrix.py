import scipy.sparse as sparse
import numpy as np


"""
Class to provide SparseMatrix capability
"""


class SparseMatrix:
    def __init__(self, dimension):
        # dimensions of the matrix (NxN)
        self.dimension = dimension
        # list of data
        self.data = []
        # list of row-numbers
        self.row = []
        # list of column-numbers
        self.col = []

    # insert entries into an existing lists for the csr-matrix
    def insert(self, entry):
        # extend data-list with the first row of the numpy array entry
        self.data.extend(entry[0, :])
        # extend row-list with the second row of the numpy array entry
        self.row.extend(entry[1, :])
        # extend col-list with the third row of the numpy array entry
        self.col.extend(entry[2, :])

    # function to return a csr-matrix build from separate lists for data, rows and colums
    def getmatrix(self):
        # create a csr sparse matrix with the dimensions (NxN)
        matrix = sparse.csr_matrix((self.data, (self.row, self.col)), shape=(self.dimension, self.dimension))
        return matrix
