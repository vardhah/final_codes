import numpy as np 
import tensorflow as tf

class AVF_search():
    def __init__(self):
        self.new_model = tf.keras.models.load_model('./DATA/saved_model/my_model')
        
    def avf_predictor(self, number,episode):
        samples_per_iteration=number
        random_seed=np.random.randint(1,100000)
        np.random.seed(random_seed)
        candidate_initial_speed=[]
        friction_of_patch=[]
        size_of_patch=[]
        location_of_patch=[]

        for i in range(samples_per_iteration):
           candidate_initial_speed.append(np.random.normal(35,9))
           friction=np.random.normal(0.6,0.10)
           if friction<=0 : friction=0 ;
           friction_of_patch.append(friction)
           size=np.random.normal(15,5)
           if size<0 : size=0
           size_of_patch.append(size)
           location_of_patch.append(np.random.normal(50,12))



        candidate_initial_speed=np.array(candidate_initial_speed).reshape(-1,1)
        friction_of_patch=np.array(friction_of_patch).reshape(-1,1)
        size_of_patch=np.array(size_of_patch).reshape(-1,1)
        location_of_patch=np.array(location_of_patch).reshape(-1,1)

        Episode=episode*np.ones((samples_per_iteration,1)).reshape(-1,1)
        data=np.concatenate((Episode,candidate_initial_speed,friction_of_patch,location_of_patch,size_of_patch),axis=1)
        container = np.load('./DATA/stdmean.npz')
        data_normalised=np.divide(np.subtract(data,container['mean']),container['std'])
        predicted_y=self.new_model.predict(data_normalised,batch_size=8)
        max_value=np.amax(predicted_y)
        array_position=np.where(predicted_y==max_value)[0]
        data_final=np.concatenate((data[array_position,:],predicted_y[array_position,:]),axis=1)
        return data_final


