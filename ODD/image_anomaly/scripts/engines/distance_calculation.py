import sys
import os
import cv2
import carla
from carla import Image
import numpy as np
from scripts.perception.perception_net import PerceptionNet
import torch


class DistanceCalculation():
    def __init__(self, ego_vehicle, leading_vehicle, perception=None):
        self.ego_vehicle = ego_vehicle
        self.leading_vehicle = leading_vehicle
        self.perception = perception
        if perception is not None:
            self.model = PerceptionNet()
            self.model = self.model.to('cuda')
            self.model.load_state_dict(torch.load(os.path.join(perception, 'perception.pt')))
            print("load the perception model successfully")


    def getTrueDistance(self):
        distance = self.leading_vehicle.get_location().y - self.ego_vehicle.get_location().y \
                - self.ego_vehicle.bounding_box.extent.y - self.leading_vehicle.bounding_box.extent.y
        return distance 
    
    def getRegressionDistance(self, image):
        if self.perception is not None:
            img = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            img = np.reshape(img, (image.height, image.width, 4))
            img = cv2.resize(img, (224,224))[:,:,:3].astype(np.float)
            img = img[:, :, ::-1]/255.0
            img = np.rollaxis(img, 2, 0)
            img = np.expand_dims(img,axis=0)
            img = torch.from_numpy(img).to(device="cuda", dtype=torch.float)
            distance = self.model(img)*100.0
            return float(distance.item())
        return None
        