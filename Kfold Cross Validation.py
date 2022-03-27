# implementation of cross fold validation.

import numpy as np

class Kfold:
    """K Fold Cross Validation with some custom functions in it."""
    def __init__(self, k_splits=None, shuffle=False):
        self.split =k_splits
        self.shuffle = shuffle
        
    def splits(self, dataset):
        "Here, dataset is splitted into train and test in K-Splits."
        self.dataset =dataset
        indexes = list(self.dataset.index)
        if self.shuffle == True:
            self.dataset =self.dataset.sample(frac=1)
        fold_size = int(len(indexes)/self.split)
        fold_split_into_train_test=[]
        pointer=0
        for split in range(self.split):
            test = indexes[pointer*fold_size : (pointer+1)*fold_size]
            if split ==0:
                train =indexes[(pointer+1)*fold_size:]
            elif split == self.split-1:
                train = indexes[0:pointer*fold_size]
            else:
                train = indexes[0:pointer*fold_size]+ indexes[(pointer+1)*fold_size:]
            fold_split_into_train_test.append((np.array(test),np.array(train)))
            pointer+=1
        return fold_split_into_train_test
