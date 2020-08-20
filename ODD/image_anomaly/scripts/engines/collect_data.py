import os
import sys
import csv
from carla import Image

class collectData():
    def __init__(self, path, isPerception):
        self.path = path
        self.isPerception = isPerception
        self.count = 0
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        csv_file = os.path.join(self.path, "label.csv")
        self.f = open(csv_file, 'w')
        self.csv_writer = csv.writer(self.f)
        if self.isPerception:
            self.csv_writer.writerow(["image_path", "true_dist", "predicted_dist", "velocity", "brake", "precipitation"])
        else:
            self.csv_writer.writerow(["image_path", "true_dist", "velocity", "brake", "precipitation"])

    def __call__(self, image, gt_distance, velocity, brake, precipitation, timestamp, regression_distance=0):
        if brake == -1:
            file_path = os.path.join(self.path, "_"+str(self.count))
            self.count+=1
        else:
            file_path = os.path.join(self.path, str(timestamp))
        image.save_to_disk(file_path)
        if self.isPerception:
            self.csv_writer.writerow([file_path, round(gt_distance, 4), round(regression_distance, 4), round(velocity, 4), round(brake, 4), round(precipitation, 4)])
        else:
            self.csv_writer.writerow([file_path, round(gt_distance, 4), round(velocity, 4), round(brake, 4), round(precipitation, 4)])
    
    def close_csv(self):
        self.f.close()