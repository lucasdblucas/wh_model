import torch
import os
import tqdm

from torch.utils.data import DataLoader, Sampler
from torchvision import transforms
from data.leprosy import Leprosy
from data.silhouettes import Silhouettes
from data.isic import Isic
from data.leprosy_isic import LeprosyISIC
from data.leprosy_fusion import LeprosyFusion
from utils.utils_data import get_transform_compose

def silhouettes_data_loader(root, batch_size=64, shuffle=True, num_workers=4):
    
    transform = transforms.Compose([
        transforms.Resize((448, 448)),
        transforms.ToTensor(),
    ])

    dataset = Silhouettes(path_to_csv=root, transform=transform)
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_mean_and_std(config, calculate_norm=False):
    
    data_config = config.CONFIG["current_project_info"]["data_config"]
    general_config = config.CONFIG["general_info"]
    # leprosy_isic_config = data_config["leprosy_isic_config"]
    
    if data_config["file_type"]["use"] == "csv":
        dir_file_type = "csvs"
    
    data_index = None
    data_loader = None
    data_path = os.path.join(general_config["data_path"]["use"], dir_file_type, data_config["data_file_name"]["use"])
    
    #* In the mean and std calculating, there is no augment transforms
    transform = get_transform_compose(config=config, calculate_norm=calculate_norm)["pytorch_compose"]
    
    if "info_regions" in data_config["data_file_name"]["use"]:
        dataset = Leprosy(
            path_to_csv=data_path, 
            transform=transform
        )
        
    elif "leprosy_isic" in data_config["data_file_name"]["use"]:
        data_dict = {
            "original_data_transforms": transform,
            "augment_data_transforms": None,
            "oversample_minority_class": {
                "use": False,
                "times": None   
            }
        }
        dataset = LeprosyISIC(
            path_to_csv=data_path,
            data_dict=data_dict
        )
        data_index = dataset.data_index
        
    elif "isic" in data_config["data_file_name"]["use"]:
        dataset = Isic(
            path_to_csv=data_path, 
            ori_transform=transform
        )
        data_index = dataset.data_index
    
    elif "info_only_regions" in data_config["data_file_name"]["use"]:
        data_dict = {
            "original_data_transforms": transform,
            "augmented_data_transforms": None,
            "oversample_minority_class": {
                "use": False,
                "times": None
            },
            "metadata_features_config": {
                "features_list": [],
                "drop_null": False
            },
            "save_info": {
                "path": None,
                "reverse_transforms": None
            }
        }
        dataset = LeprosyFusion(
            path_to_csv=data_path,
            data_dict=data_dict
        )
        data_index = dataset.all_set_index_labels
    
    if "silhouettes" in data_config["data_file_name"]["use"]:
        data_dict = {
            "original_data_transforms": transform,
            "augment_data_transforms": {
                "augmentation_flag": False,
                "augmentation_compose_dict": None,
                "augment_size": None
            },
            "normalize_output": {
                "use": False,
                "to_norm": None
            },
            "save_info": {
                "save_path": None,
                "reverse_transforms_compose": None
            }            
        }
        
        dataset = Silhouettes(
            path_to_csv=data_path, 
            data_dict=data_dict
        )
        data_index = dataset.ids
    
    data_loader = DataLoader(
        dataset, 
        batch_size=data_config["data_batch_size"]["use"], 
        num_workers=data_config["num_workers"]["use"],
        sampler=data_index
    )
    
    print("[DATA] Data for Mean and Std Calculation")
    print("Loader size: {}".format(len(data_loader)))
    print("Dataset size: {}".format(len(data_loader.dataset)))
    
    ##
    # Tensor.size(dim=None) -> Returns the size of the self tensor. If dim is not specified, the returned value is a torch.Size, a subclass of tuple. If dim is specified, returns an int holding the size of that dimension.
    # Tensor.view(*shape) → Tensor -> Returns a new tensor with the same data as the self tensor but of a different shape.
    # torch.cat(tensors, dim=0, *, out=None) → Tensor -> Concatenates the given sequence of seq tensors in the given dimension. All tensors must either have the same shape (except in the concatenating dimension) or be empty. dim (int, optional) – the dimension over which the tensors are concatenated.
    ##
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    total_batches = len(data_loader)
    total_images = len(data_loader.dataset)
    bar_range = range(total_batches)
    print(config.LINE + "[MEAN and STD] Calculating" + config.LINE)
    with tqdm.tqdm(bar_range, unit="batchs", position=0) as bar:
        for batch_idx, (data, _, _) in enumerate(data_loader):
            
            bar.set_description("Bach {}/{} - Images {}/{}".format(batch_idx + 1, total_batches, (batch_idx + 1) * len(data), total_images))
            
            # Mean over batch, height and width, but not over the channels.
            # dim = 1 would be the channels
            channels_sum += torch.mean(data, dim=[0,2,3])
            channels_squared_sum += torch.mean(data**2, dim=[0,2,3])
            num_batches += 1
            
            bar.update(1)
        
    mean = channels_sum / num_batches
    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    
    print("[MEAN and STD] mean: {}, std: {}\n".format(mean, std))
    
    return mean.numpy(force=True).tolist(), std.numpy(force=True).tolist()
