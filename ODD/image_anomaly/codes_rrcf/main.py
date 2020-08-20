import numpy as np
import pandas as pd
from rcf import RCTree
import matplotlib.pyplot as plt
import pickle
import time

train_data=pd.read_csv('train_data_small.csv')
test_data90=pd.read_csv('test_data_ppt90.csv')
test_data70=pd.read_csv('test_data_ppt70.csv')
test_data50=pd.read_csv('test_data_ppt50.csv')
test_data30=pd.read_csv('test_data_ppt30.csv')
test_10pix=pd.read_csv('test_data_ppt70.csv')
test_5pix=pd.read_csv('test_data_ppt50.csv')

print("Data loaded")
train_np_data=train_data.values


X=train_data.values[0:125]
Y=test_data90.values
Z=test_data70.values
A=test_data50.values
B=test_data30.values

print("Shape of X:",X.shape)
print("Shape of Y:",Y.shape)


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



avg_codisp = pd.Series(0.0, index=np.arange(n))
index = np.zeros(n)
for tree in forest: 
    codisp = pd.Series({leaf : tree.codisp(leaf)
                       for leaf in tree.leaves})
    avg_codisp[codisp.index] += codisp
    np.add.at(index, codisp.index.values, 1)
avg_codisp /= index
print("Length of average coDisp:",len(avg_codisp))

#plt.plot(avg_codisp)
#plt.hlines(25,1,1100,'r')
#plt.grid()
#plt.show()
print("Max CoDisp: ",avg_codisp.max())
#print('Avg coDisp:',avg_codisp)

#for all test points:
#t_cnt=0
#num_test_points=Y.shape[0]


disp_val=avg_codisp.tolist()
for i in range(5):
 new_testpoint=Y[(i)]
 point_codisp=0
 start=time.time()
 for tree in forest:
    tree.insert_point(new_testpoint,index=3986+i)
 end=time.time()
 print("time taken:",start-end)
 for tree in forest: 
    codisp =  tree.codisp(3986+i)                   
    point_codisp += codisp
 point_codisp=point_codisp/len(forest)
 disp_val.append(point_codisp)
 print('-->Test point coDisp:',point_codisp)
 start=time.time()
 for tree in forest: 
  tree.forget_point(index=3986+i)
 end=time.time()
 print("time taken:",start-end)



#for new training points
cnt=0
cntt=0
for i in range(30):
 new_testpoint=train_np_data[(500+i)]
 point_codisp=0
 ttrees=0
 for tree in forest:
    tree.insert_point(new_testpoint,index=5986+i)
    ttrees+=1
 for tree in forest: 
    codisp =  tree.codisp(5986+i)                   
    point_codisp += codisp
 point_codisp=point_codisp/len(forest)
 disp_val.append(point_codisp)
 print('-->New point coDisp:',point_codisp)
 if point_codisp<avg_codisp.max():
    for tree in forest: 
     tree.forget_point(index=5986+i)
    print("removed point")
    cnt+=1 
 else:
    print("point accepted in tree") 
    cntt+=1   
print("Removed points:",cnt)  
print("Accepted points:",cntt)

#np.savetxt('codisp.csv',delimiter=',') 


for i in range(5):
 new_testpoint=Z[(i)]
 point_codisp=0
 for tree in forest:
    tree.insert_point(new_testpoint,index=7986+i)
 for tree in forest: 
    codisp =  tree.codisp(7986+i)                   
    point_codisp += codisp
 point_codisp=point_codisp/len(forest)
 disp_val.append(point_codisp)
 print('-->Test point coDisp:',point_codisp)
 for tree in forest: 
   tree.forget_point(index=7986+i)


for i in range(30):
 new_testpoint=train_np_data[(600+i)]
 point_codisp=0
 ttrees=0
 for tree in forest:
    tree.insert_point(new_testpoint,index=9986+i)
    ttrees+=1
 for tree in forest: 
    codisp =  tree.codisp(9986+i)                   
    point_codisp += codisp
 point_codisp=point_codisp/len(forest)
 disp_val.append(point_codisp)
 print('-->New point coDisp:',point_codisp)
 if point_codisp<avg_codisp.max():
    for tree in forest: 
      tree.forget_point(index=9986+i)
    print("removed point")
    cnt+=1 
 else:
    print("point accepted in tree") 
    cntt+=1   


for i in range(5):
 new_testpoint=A[(i)]
 point_codisp=0
 for tree in forest:
    tree.insert_point(new_testpoint,index=4986+i)
 for tree in forest: 
    codisp =  tree.codisp(4986+i)                   
    point_codisp += codisp
 point_codisp=point_codisp/len(forest)
 disp_val.append(point_codisp)
 print('-->Test point coDisp:',point_codisp)
 for tree in forest: 
   tree.forget_point(index=4986+i)


for i in range(30):
 new_testpoint=train_np_data[(700+i)]
 point_codisp=0
 ttrees=0
 for tree in forest:
    tree.insert_point(new_testpoint,index=6986+i)
    ttrees+=1
 for tree in forest: 
    codisp =  tree.codisp(6986+i)                   
    point_codisp += codisp
 point_codisp=point_codisp/len(forest)
 disp_val.append(point_codisp)
 print('-->New point coDisp:',point_codisp)
 if point_codisp<avg_codisp.max():
    for tree in forest: 
      tree.forget_point(index=6986+i)
    print("removed point")
    cnt+=1 
 else:
    print("point accepted in tree") 
    cntt+=1   



disp_val=np.array(disp_val)
np.savetxt('Disp_value.csv', disp_val, delimiter=',')


plt.plot(disp_val)
plt.grid()
plt.title('Displacement Value on stream of image data')
plt.xlabel('image_frame_number')
plt.ylabel('Disp Value')
plt.show()
