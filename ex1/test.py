import numpy as np

a = np.array([[1, 2, 3, 4, 5], 
                [6, 7, 8, 9, 10],[11,12,13,14,15]]) 
      
# applying ndarray.__irshift__() method 
b = np.zeros(a.shape,dtype = a.dtype)
b[:,1:] = a[:,:-1]
#b[:,0] = np.zeros((0,b.shape[1]))
print(a) 
print(b)

print(np.ones((5,5)))