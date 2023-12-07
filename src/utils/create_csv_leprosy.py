import pandas as pd 
import os 

csv_origin = "C:/Users/lucas/Desktop/Datasets_in_use/info_regions.csv"
csv_adapted = "C:/Users/lucas/Desktop/Datasets_in_use"
new_csv_name = "info_regions_02.csv"

dataframe = pd.read_csv(csv_origin, sep=',')

def returnedString(str_ret):
    
    split_str = str_ret.split("\\")
    
    # print(split_str)
    
    return os.path.join(csv_adapted, "BDHansen_Local", split_str[-2], split_str[-1])

dataframe["image_path"] = dataframe["image_path"].apply(returnedString)

dataframe.to_csv(os.path.join(csv_adapted, new_csv_name))

        