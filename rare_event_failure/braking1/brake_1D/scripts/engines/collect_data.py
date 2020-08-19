import os
import sys
import csv

class collectData():
    def __init__(self, path):
        self.path = path
        self.count = 0
        data_file='Data.csv'
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        csv_file = os.path.join(self.path, data_file)
        self.f = open(csv_file, 'w')
        self.csv_writer = csv.writer(self.f)
        
        self.csv_writer.writerow(["Episode","Kick_Speed","friction", "Rewards","Stop_Distance"])

    def __call__(self, episode,kickspd,friction_of_patch,reward, gt_distance):
        
        self.csv_writer.writerow([episode, round(kickspd, 2),round(friction_of_patch, 2),round(reward, 2), round(gt_distance, 2)])
    
    def close_csv(self):
        self.f.close()
