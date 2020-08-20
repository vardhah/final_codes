import numpy as np
import pandas as pd
from rcf import RCTree
import matplotlib.pyplot as plt
import pickle
import dill
import time

train_data=pd.read_csv('train_data_small.csv')

print("Data loaded")
train_np_data=train_data.values


X=train_data.values[0:100]


print("Shape of X:",X.shape)


n=X.shape[0]
d=X.shape[1]
print("n is:",n)
print("d is:",d)

i=0
forest = []
# Specify forest parameters
num_trees = 100
tree_size = 64
sample_size_range = (n//tree_size,tree_size)

while len(forest) < num_trees:
    # Select random subsets of points uniformly from point set
    ixs = np.random.choice(n, size=sample_size_range,
                           replace=False)
    # Add sampled trees to forest
    trees = [RCTree(X[ix], index_labels=ix)
             for ix in ixs]
    forest.extend(trees)
    print("created forest", i)
    i+=1
    print("length of forest:",len(forest))



avg_disp = pd.Series(0.0, index=np.arange(n))
index = np.zeros(n)
for tree in forest: 
    disp = pd.Series({leaf : tree.disp(leaf)
                       for leaf in tree.leaves})
    avg_disp[disp.index] += disp
    np.add.at(index, disp.index.values, 1)
avg_disp /= index
print("Length of average disp:",len(avg_disp))

#plt.plot(avg_disp)
#plt.hlines(25,1,1100,'r')
#plt.grid()
#plt.show()
print("Max disp: ",avg_disp.max())
#print('Avg disp:',avg_disp)

#for all test points:
#t_cnt=0
#num_test_points=Y.shape[0]
peak_val=avg_disp.max()

disp_val=avg_disp.tolist()


#for new training points
cnt=0
cntt=0
for i in range(890):
 new_testpoint=train_np_data[(100+i)]
 point_disp=0
 ttrees=0
 for tree in forest:
    tree.insert_point(new_testpoint,index=5986+i)
    ttrees+=1
 for tree in forest: 
    disp =  tree.disp(5986+i)                   
    point_disp += disp
 point_disp=point_disp/len(forest)
 disp_val.append(point_disp)
 print('-->New point disp:',point_disp)
 if point_disp<peak_val:
    for tree in forest: 
     tree.forget_point(index=5986+i)
    print("removed point")
    cnt+=1 
 else:
    print("point accepted in tree") 
    peak_val=point_disp
    cntt+=1   
print("Removed points:",cnt)  
print("Accepted points:",cntt)

 




plt.plot(disp_val)
plt.grid()
plt.title('Displacement Value on stream of image data')
plt.xlabel('image_frame_number')
plt.ylabel('Disp Value')
plt.show()

disp_val=np.array(disp_val)
np.savetxt('Disp_value.csv', disp_val, delimiter=',')
print("written Disp value") 
with open('trained_model', 'wb') as output:
  dill.dump(forest, output, pickle.HIGHEST_PROTOCOL)
print("finished")