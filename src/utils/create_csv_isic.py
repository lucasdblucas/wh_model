import pandas as pd 
import os 

csv_origin = "C:/Users/lucas/Desktop/Datasets_in_use/isic\isic_images_info.csv"
csv_adapted = "C:/Users/lucas/Desktop/Datasets_in_use/isic"
new_csv_name = "isic_images_info_02.csv"

dataframe = pd.read_csv(csv_origin, sep=',')

def returnedString(str_ret):
    
    return os.path.join(csv_adapted, "data", str_ret.split("\\")[-1])

dataframe["image_path"] = dataframe["image_path"].apply(returnedString)

dataframe.to_csv(os.path.join(csv_adapted, new_csv_name))

        