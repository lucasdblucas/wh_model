import pandas as pd 
import os 

images_path = "C:/Users/lucas/GoogleDrive/Meu Drive/Colab Notebooks/Segmentação/segmented_images/biotonix/separate_images"
path_o = "C:/Users/lucas/GoogleDrive/Meu Drive/Colab Notebooks/dataset/dataset_biotonix/dataset_biotonix_mod_02.csv"
save_path = "C:/Users/lucas/Documents/Area_de_Trabalho/framework-models/src/data/csvs/silhouettes_1.csv"

files_names = []
for folder, _, files in os.walk(images_path):
    for key, f in enumerate(files):
        name = f.split("}")[1]
        name = name.split(".")[0]
        
        if "03" in name or "FA" in name:
            files_names.append(f)        

dataframe = pd.read_csv(path_o, sep=',')

df_of_interest = pd.DataFrame(dataframe[["id", "height_cm", "weight_kg"]])

def returnedString(str_ret):
    # return os.path.join(images_path, "".join([str_ret[:-6], '_group_2h.jpg']))
    for name in files_names:
        if str_ret[:-6] in name:
            # print(str_ret)
            return os.path.join(images_path, name)

df_of_interest.insert(3, "path_to_images", dataframe['filename'].apply(returnedString))

df_of_interest.to_csv(save_path)

df_of_interest.tail(10)

        