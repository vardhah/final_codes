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
        
        #self.csv_writer.writerow(["Episode","kickSpeed1","dist_o1", "vel1","action1","kickSpeed2","loc2",
         #"dis_o2","vel2","action2"])
        self.csv_writer.writerow(["dist_o1", "vel1","mu_l","action","shift_obstacle"])

    #def __call__(self, episode,ks1,d1,v1,a1,ks2,loc2,d2,v2,a2):
    def __call__(self,d1,v1,mu1,a,shift):
        self.csv_writer.writerow([round(d1, 2),round(v1, 2),round(mu1, 2),round(a, 2),round(shift, 2)])
        #self.csv_writer.writerow([episode,round(ks1, 2),round(d1, 2),round(v1, 2),round(a1, 2),
         #round(ks2, 2),round(loc2, 2),round(d2, 2),round(v2, 2),round(a2, 2)])

    
    def close_csv(self):
        self.f.close()
