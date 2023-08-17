import torch.nn as nn
import numpy as np
from statistics import variance
class lsqScale(nn.Module):
    def __init__(self):
        super(lsqScale, self).__init__()
        
    def fit(self,input):
        x=np.arange(0,len(input))
        A = np.vstack([x, np.ones(len(x))]).T
        self.weight = np.linalg.lstsq(A, input, rcond=None)[0]
        # mean=np.matmul(A,self.weight)
        self.variance=np.var(input,axis=0)
        self.std=np.sqrt(self.variance)
    def transform(self,input):
        x=np.arange(0,len(input))
        A = np.vstack([x, np.ones(len(x))]).T
        mean=np.matmul(A,self.weight)
        
        
        output=(input-mean)/self.std
        return output
    
    def inverse_transform(self,input):
        x=np.arange(0,len(input))
        A = np.vstack([x, np.ones(len(x))]).T
        mean=np.matmul(A,self.weight)
        output=input*self.std+mean
        return output

