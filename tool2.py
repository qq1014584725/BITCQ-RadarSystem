import os
import shutil
import csv

from torch import save


root_path = "/data1/user/WeatherRadar/RadarExtrapolation/jiangsu_dataset/"
csv_path = root_path + "Train.csv"
img_root_path = root_path + "Train/Radar/"
save_dir = "/data1/lx/jiangsu_dataset/bb/"

result_list = []
num = 1
with open(csv_path) as f:
    f_csv = csv.reader(f)
    line = next(f_csv)
    while line:
        os.mkdir(save_dir + f"sample_{num}")
        j = 1
        for index in line:
            shutil.copy(img_root_path + f"radar_{index}", save_dir + f"sample_{num}/" + f"img_{j}.png")
            j += 1
            if  j==31: break
        
        line = next(f_csv)
        num += 1
    result_list.append()
    print(result_list)