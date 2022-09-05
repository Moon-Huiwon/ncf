#%%
import numpy as np
from scipy import sparse
import scipy.sparse as sp
# %%
class Dataset(object):
    '''
    classdocs
    '''
    
    def __init__(self, path):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(R"C:\Users\문희원\Desktop\neural_collaborative\filtering-master\Data\.train.rating")
        self.testRatings = self.load_rating_file_as_list(R"C:\Users\문희원\Desktop\neural_collaborative_filtering-master\Data\.test.rating")
        self.testNegatives = self.load_negative_file(R"C:\Users\문희원\Desktop\neural_collaborative_filtering-master\Data\.test.negative")
        assert len(self.testRatings) == len(self.testNegatives)
        
        self.num_users, self.num_items = self.trainMatrix.shape
        
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList
    
    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1: ]:
                    negatives.append(int(X))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList
    def load_rating_file_as_matrix(self, filename):
        '''
        Read .rating file and Return dox matrix.
        The first line of .rating files is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()
        # Construct matrix
        mat = sp.dox_matrix((num_users+1, num_items+1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if (rating > 0):
                    mat[user, item] = 1.0
                line = f.readline()
        return mat
# %%
