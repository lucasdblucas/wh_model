import torch
import os
import torch.utils.data as data

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms
from data.silhouettes import Silhouettes
from torch.utils.data import Subset, Dataset
from configs.config import Config
from utils.utils_plots_regression import plot_and_save as plot_and_save_regr
from utils.utils_plots_regression import plot_and_save_sns_countplot
from utils.utils_data import get_transform_compose
from data.define_mean_std import get_mean_and_std
from utils.utils_json import saveorupdate_json
from errors.framework_error import FrameworkError

def loader_repr(loader):
    
    if isinstance(loader.dataset, (Dataset, Subset)) :
        s = ('{dataset.__class__.__name__} Loader: \n'
             'num_workers={num_workers}, '
             'pin_memory={pin_memory}, '
             'sampler={sampler.__class__.__name__}\n'
            #  'Root: {dataset.root}\n'
             )
        s = s.format(**loader.__dict__)
        s += 'Data Points: {}\n'
        s += 'Dataloader Points: {}\n'
        s = s.format(len(loader.dataset), len(loader))
    return s

def get_dataloader_regression(dataset, config) -> tuple:
    task_config = config.CONFIG["current_project_info"]["task_config"]
    train_infe_config = config.CONFIG["current_project_info"]["train_infe_config"]
    general_info = config.CONFIG["general_info"]
    data_config = config.CONFIG["current_project_info"]["data_config"]
    transform_config = data_config["transform_config"]
    
    dataset_size = len(dataset)
    str_error = "Duplicated indexes inside the dataset. It should not happen."
    assert not dataset.dataframe.index.duplicated().any(), str_error
    
    if data_config["split_data_type"]["use"] in data_config["split_data_type"]["choices"][1:-2]: #* train/test or train/val/test    
        
        #* Split your data into train and validation sets. only the original data should be split
        train_indices, test_indices = train_test_split(
            dataset.origin_data_index,
            train_size=data_config["train_size"]["use"],
            random_state=train_infe_config["seed"]["use"]
        )
        
        dataset.train_index = train_indices
        dataset.test_index = test_indices
        
        #*teste
        #* If it is executing only a test, the number of elements will be reduced ('num_elements')
        if task_config["testing_config"]["use"]:
            train_indices = train_indices[:task_config["testing_config"]["num_elements"]["use"]]
            test_indices = test_indices[:task_config["testing_config"]["num_elements"]["use"]]
            
        
        test_sampler = SubsetRandomSampler(
            test_indices, 
            generator=torch.Generator().manual_seed(train_infe_config["seed"]["use"])
        )
        if data_config["split_data_type"]["use"] == data_config["split_data_type"]["choices"][1]: #*train/test
            # data_config["val_size"]["use"] = 0.1
            # data_config["train_size"]["use"] -= data_config["val_size"]["use"]
            
            dataset.actions_after_split()
            
            train_sampler = SubsetRandomSampler(
                train_indices, 
                generator=torch.Generator().manual_seed(train_infe_config["seed"]["use"])
            )
            
            train_dataloader = DataLoader(
                dataset=dataset, 
                batch_size=data_config["data_batch_size"]["use"], 
                sampler=train_sampler, 
                pin_memory=True, 
                # num_workers=data_config["num_workers"]["use"]
            )
            test_dataloader = DataLoader(
                dataset=dataset, 
                batch_size=data_config["data_batch_size"]["use"], 
                sampler=test_sampler, 
                pin_memory=True, 
                # num_workers=data_config["num_workers"]["use"]
            )
                
            #* save plot for distribution
            save_dir = os.path.join(general_info["save_model_path"]["use"], train_infe_config["save_directory"]["use"])
            distribution_dict = {
                "all": dataset.get_distribution(list_index=list(range(dataset_size))),
                "train": dataset.get_distribution(list_index=train_indices),
                "test": dataset.get_distribution(list_index=test_indices)
            }
            # plot_and_save_regr(save_dir, targets_over_sets=distribution_dict)
            plot_and_save_sns_countplot(save_dir, targets_over_sets=distribution_dict)
            
            return train_dataloader, test_dataloader
    
        elif data_config["split_data_type"]["use"] == data_config["split_data_type"]["choices"][2]: #* train/val/test
        
            train_indices, val_indices = train_test_split(
                train_indices, 
                test_size=data_config["val_size"]["use"], 
                random_state=train_infe_config["seed"]["use"]
                )
            
            train_sampler = SubsetRandomSampler(train_indices, generator=torch.Generator().manual_seed(train_infe_config["seed"]["use"]))
            val_sampler = SubsetRandomSampler(val_indices, generator=torch.Generator().manual_seed(train_infe_config["seed"]["use"]))
            
            # train_dataloader = DataLoader(dataset=dataset, batch_size=data_config["data_batch_size"]["use"], sampler=train_sampler, pin_memory=True, num_workers=data_config["num_workers"]["use"])
            # val_dataloader = DataLoader(dataset=dataset, batch_size=data_config["data_batch_size"]["use"], sampler=val_sampler, pin_memory=True, num_workers=data_config["num_workers"]["use"])
            # test_dataloader = DataLoader(dataset=dataset, batch_size=data_config["data_batch_size"]["use"], sampler=test_sampler, pin_memory=True, num_workers=data_config["num_workers"]["use"])
                
            #* save plot for distribution
            save_dir = os.path.join(general_info["save_model_path"]["use"], train_infe_config["save_directory"]["use"])
            distribution_dict = {
                "all": dataset.get_distribution(list_index=list(range(dataset_size))),
                "train": dataset.get_distribution(list_index=train_indices),
                "val": dataset.get_distribution(list_index=val_indices),
                "test": dataset.get_distribution(list_index=test_indices)
            }
            plot_and_save_regr(save_dir, targets_over_sets=distribution_dict)
            
            return train_dataloader, val_dataloader, test_dataloader
    
    elif data_config["split_data_type"]["use"] in data_config["split_data_type"]["choices"][0]: #* train
        
        # train_dataloader = DataLoader(dataset, batch_size=data_config["data_batch_size"]["use"], shuffle=True, pin_memory=True, num_workers=data_config["num_workers"]["use"])
        
        return train_dataloader
    
    elif data_config["split_data_type"]["use"] in data_config["split_data_type"]["choices"][-2]: #* inference
        
        _, infe_indices = train_test_split(
            list(range(dataset_size)),
            test_size=data_config["infe_size"]["use"],
            random_state=train_infe_config["seed"]["use"]
        )
        
        infe_sampler = SubsetRandomSampler(infe_indices, generator=torch.Generator().manual_seed(train_infe_config["seed"]["use"]))
        
        # pred_dataloader = DataLoader(dataset=dataset, batch_size=data_config["data_batch_size"]["use"], sampler=infe_sampler, pin_memory=True, num_workers=data_config["num_workers"]["use"])
        
        return pred_dataloader

def silhouettes_loader(config: Config = None) -> tuple:
    
    #* Config
    train_config = config.CONFIG["current_project_info"]["train_infe_config"]
    data_config = config.CONFIG["current_project_info"]["data_config"]
    transform_config = data_config["transform_config"]
    general_config = config.CONFIG["general_info"]
    json_path = os.path.join(general_config["save_model_path"]["use"], os.path.basename(config.PATH_JSON_CONFIG))
    
    #* reduced name for mean and standard deviation
    mean = None
    std = None
        
    #* transforms    
    if transform_config["normalize_input"]["use"]:    
        if not transform_config["normalize_input"]["name"] in general_config["mean"].keys() or not transform_config["normalize_input"]["name"] in general_config["std"].keys():
            mean, std = get_mean_and_std(config=config, calculate_norm=True)
            
            general_config["mean"][transform_config["normalize_input"]["name"]] = mean
            general_config["std"][transform_config["normalize_input"]["name"]] =  std
            
            #* save new info
            general_config_tosave = {}
            general_config_tosave["general_info"] = general_config
            general_config_tosave["file_info"] = config.CONFIG["file_info"]
            saveorupdate_json(
                json_path=json_path, 
                config=general_config_tosave
            )
        else:
            mean = general_config["mean"][transform_config["normalize_input"]["name"]]
            std = general_config["std"][transform_config["normalize_input"]["name"]]
            
    #* the dataset type define where is it
    if data_config["file_type"]["use"] == "csv":
        dir_file_type = "csvs"
    else:
        pass
    
    #* transform for original data
    original_transforms_compose_dict = get_transform_compose(config=config)["pytorch_compose"]
    
    #* get transform for augmented data and initialize variables
    aug_transforms_compose_dict = None
    augment_size = None
    #* do augment?
    #TODO: it is not the case until now. If this dataset needs to augment size, so this part should be changed.
    if transform_config["augment"]["use"]:
        aug_transforms_compose_dict = get_transform_compose(config=config, augment=True)
        augment_size = transform_config["augment_size"]
    
    #* normalize output?
    normalize_output = True if transform_config["normalize_output"]["use"] else False
    
    data_dict = {
        "original_data_transforms": original_transforms_compose_dict,
        "augment_data_transforms": {
            "augmentation_flag": transform_config["augment"]["use"],
            "augmentation_compose_dict": aug_transforms_compose_dict,
            "augment_size": augment_size
        },
        "normalize_output": {
            "use": normalize_output,
            "to_norm": transform_config["normalize_output"]["to_norm"]
        },
        "save_info": {
            "save_path": os.path.join(general_config["save_model_path"]["use"], train_config["save_directory"]["use"]),
            "reverse_transforms_compose": transforms.Compose([
                #TODO: in this case, it is considering a gray image. If a RGB image is the entry, so this should be changed.
                transforms.Normalize(
                    mean=(-mean[0]/std[0]),
                    std=(1/std[0])
                ),
                transforms.ToPILImage()
            ])
        }
    }
    
    dataset = Silhouettes(
        path_to_csv=os.path.join(general_config["data_path"]["use"], dir_file_type, data_config["data_file_name"]["use"]),
        data_dict=data_dict
    )
    
    return get_dataloader_regression(dataset=dataset, config=config)