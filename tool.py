import os
import shutil


# for i in range(1, 113):
#     if not os.path.exists("/data1/lx/jiangsu_dataset/validation/sample_" + str(i)):
#         os.makedirs("/data1/lx/jiangsu_dataset/validation/sample_" + str(i))
        

path_list = os.listdir("/data1/user/WeatherRadar/RadarExtrapolation/jiangsu_dataset/TestA/Radar/")
path_list.sort(key=lambda x:x[6:11])

for i in range(1, 113):
    for j in range(1, 31):
        file_name = path_list.pop(0)
        shutil.copy("/data1/user/WeatherRadar/RadarExtrapolation/jiangsu_dataset/TestA/Radar/"+file_name, "/data1/lx/jiangsu_dataset/validation/sample_" + str(i))
        os.rename("/data1/lx/jiangsu_dataset/validation/sample_" + str(i) + "/" + file_name, "/data1/lx/jiangsu_dataset/validation/sample_" + str(i) + "/" + "img_{}.png".format(j));