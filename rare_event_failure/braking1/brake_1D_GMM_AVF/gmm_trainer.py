from sklearn import mixture
import pickle
import numpy as np

class gmm_trainer():
    def __init__(self,failed_data):
      
      self.fails = failed_data
      

    def training(self, failure):
        self.fails=np.append(self.fails,failure)
        #print(self.fails.shape)
        self.fails=np.transpose(self.fails.reshape(1,-1))
        #print(self.fails.shape)
        print("-------------Retraining---------------------------")
        #print(self.fails)
        gmm = mixture.GaussianMixture(n_components=2, covariance_type='full',n_init=200).fit(self.fails)
        with open('gmm_1D', 'wb') as f:
           pickle.dump(gmm, f)
           print("---->>>>Trained and Stored")
        return 0


