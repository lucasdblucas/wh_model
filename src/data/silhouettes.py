import pandas as pd
import os
import torch 
import numpy as np

from torchvision import transforms
from datetime import datetime
from PIL import Image
from torch.utils.data import Dataset


class Silhouettes(Dataset):
    def __init__(self, path_to_csv, data_dict):
        #* load csv file
        dataframe = pd.read_csv(path_to_csv, sep=',', index_col=0)
        #* add 'transform' column
        dataframe["transform"] = "original_data"
        #TODO: Atention
        self.orig_transforms = data_dict["original_data_transforms"]
        self.aug_transforms = data_dict["augment_data_transforms"]
        self.aug_flag = data_dict["augment_data_transforms"]["augmentation_flag"]
        
        #* do output normalization?
        self.overall_mean = None
        self.overall_std = None
        if data_dict["normalize_output"]["use"]:    
            t_weight = dataframe["weight_kg"].values
            t_height = dataframe["height_cm"].values
            aux = t_weight
            aux += t_height 
            if "weight" in data_dict["normalize_output"]["to_norm"] and "height" in data_dict["normalize_output"]["to_norm"]:
                self.overall_mean = np.mean(aux)
                self.overall_std = np.std(aux)
            elif "weight" in data_dict["normalize_output"]["to_norm"] and "height" not in data_dict["normalize_output"]["to_norm"]:
                self.overall_mean = np.mean(t_weight)
                self.overall_std = np.std(t_weight)
            elif "weight" not in data_dict["normalize_output"]["to_norm"] and "height" in data_dict["normalize_output"]["to_norm"]:
                self.overall_mean = np.mean(t_height)
                self.overall_std = np.std(t_height)
        
        #* define augmentation data considering the augmentation size, the number of images to augment
        #* The augmentation should not be here. The augmented data is for train data e should not be in test data, so the augmentation
        #* should be done after the data split.
        # print("TESTE: flag: {}".format(data_dict["augment_data_transforms"]["augmentation_flag"]))
        if data_dict["augment_data_transforms"]["augmentation_flag"]:
            # dataframe = self.augmented_dataset(
            #     dataframe_a=dataframe, 
            #     augment_size=data_dict["augment_data_transforms_info"]["augment_size"]
            # )
            pass
        #* original data and augmentated data indexes
        self.origin_data_index = dataframe.loc[dataframe["transform"] == "original_data"].index.to_list()
        self.aug_data_index = dataframe.loc[dataframe["transform"] == "augmented_data"].index.to_list()
        
        #* add 'bmi' column
        #TODO: Add a condition to make this column. It is not all the time that bmi will be required.
        dataframe["bmi"] = dataframe["weight_kg"] / ((dataframe["height_cm"] / 100) ** 2)
        
        #* list of index, witch are the individuals IDs
        self.ids = dataframe.index.tolist()
        #* List of target values - (weight, height, bmi)
        self.targets_name = ["weight_kg", "height_cm", "bmi"]
        # self.targets = list(zip(dataframe["weight_kg"].values, dataframe["height_cm"].values, dataframe["bmi"].values)) 
        # #* transform a tuple list in a list of lists. The anterior 'zip' function returns tuple values.
        # self.targets = [[w, h, b] for w, h, b in self.targets]
        #* image paths
        self.path_to_images = dataframe["path_to_images"]  # List of image names
        #* transforms - for original data or augmented data
        self.transforms_to_apply = dataframe["transform"] # List of transform type - (A or B)
        
        #* Sample size
        self.dataset_len = len(self.ids)
        self.dataframe = dataframe
        self.train_index = None
        self.test_index = None
        self.val_index = None
        
        #* select randomly image samples to save
        # pd.np.random.seed(datetime.now().second)
        self.images_to_save = dataframe.loc[dataframe["transform"] == "original_data"].sample(n=10).index.tolist()
        if len(self.aug_data_index) > 0:
            self.images_to_save += dataframe.loc[dataframe["transform"] == "augmented_data"].sample(n=10).index.tolist()
        self.save_path = data_dict["save_info"]["save_path"]
        
        self.reverse_transform = data_dict["save_info"]["reverse_transforms_compose"]
        
    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        path_to_image = self.path_to_images.loc[idx]
        image_name = os.path.basename(path_to_image)
        original_image = Image.open(path_to_image)
        target = np.array(self.dataframe.loc[idx, self.targets_name].values, dtype=float)
        transform_type = self.transforms_to_apply.loc[idx]
        target_image = None
        
        #* transform to original images
        if transform_type == "original_data":
            target_image = self.orig_transforms(original_image.copy())
            if idx in self.images_to_save and self.save_path != None:
                #* before transformations
                self.store_image_if_not_saved(
                    image=original_image,
                    image_name="original_before_" + image_name
                )
                #* after transformations
                self.store_image_if_not_saved(
                    image=self.reverse_transform(target_image),
                    image_name="original_after_" + image_name
                )
        #* transform to augmented images
        elif transform_type == "augmented_data":
            target_image = self.aug_transforms(original_image.copy())
            if idx in self.images_to_save and self.save_path != None:
                #* before transformations
                self.store_image_if_not_saved(
                    image=original_image,
                    image_name="augmented_before_" + image_name
                )
                #* after transformations
                self.store_image_if_not_saved(
                    image=self.reverse_transform(target_image),
                    image_name="augmented_after_" + image_name
                )
                
        #* normalize outputs
        if self.overall_mean and self.overall_std:
            target = self.normalize_output(output=target)
        else:
            # target = torch.tensor(target).float()
            # print("Target: {}".format(target))
            target = torch.tensor(target).float()
        
        #TODO: idx has repeated entries
        return target_image, [path_to_image, idx], target # [weight, height, bmi]
    
    def get_distribution(self, list_index):
        return np.array([data for index, data in self.dataframe.loc[list_index, self.targets_name].iterrows()])
    
    def augmented_dataset(self, dataframe_a, augment_size):
        # dataframe_b = dataframe_a.sample(augment_size)
        # dataframe_b["transform"] = "augmented_data"
        
        # return pd.concat([dataframe_a, dataframe_b])
        pass
    
    def normalize_output(self, output):
        
        # ret = [(output[0] - self.overall_mean)/ self.overall_std, (output[1] - self.overall_mean)/ self.overall_std]
        
        # return torch.from_numpy(np.array(ret)).float()
        
        pass
    
    def store_image_if_not_saved(self, image: Image = None, image_name: str = None) -> bool:
        
        for _, _, files in os.walk(self.save_path):
            #* save original image. In case of augmented images, before transfomrms
            #* save image, only if it is not saved already
            if image_name not in files: image.save(os.path.join(self.save_path, image_name))
            break
    
    def actions_before_split(self):
        # if self.drop_null_values:
        #     self.dataframe = self.dataframe.dropna(subset=self.list_of_features)
            
        #     self.update_info()
        pass
        
    def actions_after_split(self):
        if self.aug_flag:
            # train_indices += self.dataset.aug_data_index
            pass
        
        # if self.oversample_minority_class_flag:
        #     df = self.oversample_minority_class(
        #         train_dataframe=self.dataframe.loc[self.train_index_labels],
        #         times=self.oversample_minority_class_times
        #     )
            
        #     #* update train labels
        #     self.train_index_labels = df.index.tolist()
        #     #* increment the test labels
        #     df = pd.concat([df, self.dataframe.loc[self.test_index_labels]])
        #     #* update dataframe 
        #     self.dataframe = df
            
        #     self.update_info()            

        # str_error = "Duplicated indexes inside the dataset. It should not happen."
        # assert not self.dataframe.index.duplicated().any(), str_error
        
        pass