import numpy as np

random_mat = np.random.randint(low = 0,high = 9,size =(5,6))
print("random matrix:\n",random_mat)

new_10_loc = random_mat.argmin(1)[4]
random_mat[4,new_10_loc] = 10

print("random matrix after changing the minimal value in row #5 to 10:\n",random_mat)
