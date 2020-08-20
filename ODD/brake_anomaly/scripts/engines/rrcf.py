import numpy as np 
import pandas as pd
import rrcf
import dill
import matplotlib 
import matplotlib.pyplot as plt

class rcf():
    def __init__(self,num_tree):
        #print("in rcf")
        with open('rcf_model_10percent', 'rb') as f:
           self.forest = dill.load(f)
        print("=> loaded anomaly detector trained Model...")
        self.num_trees = num_tree
        container = np.load('./DATA/training_details_10p.npz')
        #print("Loaded container")
        print("=> Training's peak coDisp value is:",container['threshold'])
        self.n=np.asscalar(container['n'])
        self.threshold=np.asscalar(container['threshold'])
        
   
    def predictor(self,data):
        #print("New data is",data)
        #print(self.forest)
        for tree in self.forest:
          tree.insert_point(data,index=self.n)
        point_codisp = 0
        for tree in self.forest: 
          codisp = tree.codisp(self.n)
          point_codisp += codisp
        point_codisp /= self.num_trees
        return (point_codisp,self.threshold)
    
    def delete_node(self):
        for tree in self.forest:
           tree.forget_point(index=self.n) 
        #print("deleted the node")

    """
    def trainer(self):
        X=self.data[["dist_o1","vel1","mu_l"]].values
        self.n=X.shape[0]
        while len(self.forest) < self.num_trees:
          trees = rrcf.RCTree(X)
          self.forest.append(trees)
        avg_codisp = pd.Series(0.0, index=np.arange(self.n))
        index = np.zeros(self.n)
        for tree in self.forest: 
          codisp = pd.Series({leaf : tree.codisp(leaf) for leaf in tree.leaves})
        avg_codisp[codisp.index] += codisp
        np.add.at(index, codisp.index.values, 1)
        avg_codisp /= index
        self.threshold=avg_codisp.max()
        print("training done & threshold is:",self.threshold)
    """