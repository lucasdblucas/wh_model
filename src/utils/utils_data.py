from torchvision import transforms
from errors.framework_error import FrameworkError

import torchvision
torchvision.disable_beta_transforms_warning() #* desable warns about beta transformation -> v2

from torchvision.transforms import v2
import albumentations as A

def get_transform_compose(config, calculate_norm=False, augment=False) -> dict:
    data_config = config.CONFIG["current_project_info"]["data_config"]
    transform_config = data_config["transform_config"]
    general_config = config.CONFIG["general_info"]
    
    #* transform list
    pytorch_list_transforms = []
    albumentations_list_transforms = []
    
    #* augment or not
    if augment:
        #* is augmentation config settled?
        if transform_config["augment"]["use"]:
            
            #* random transforms for image augmentation - Random Augmentation
            if transform_config["augment"]["random_config"]["use"]:
                random_config = transform_config["augment"]["random_config"]
                
                #* albumentation module compose
                if random_config["coarse_drop"]["use"]:
                    albumentations_list_transforms.append(
                        A.CoarseDropout(
                            max_holes=random_config["coarse_drop"]["max_holes"],
                            max_height=random_config["coarse_drop"]["max_height"],
                            max_width=random_config["coarse_drop"]["max_width"],
                            p=random_config["coarse_drop"]["probability"]
                        )
                    )
                    
                #* pytorch module compose
                #TODO: deve-se verificar que random vertical e horizontal flip pode ter um argumento 'p' para definir a probabilidade que será aplicado.
                if random_config["random_vertical_flip"]: pytorch_list_transforms.append(transforms.RandomVerticalFlip())
                if random_config["random_horizontal_flip"]: pytorch_list_transforms.append(transforms.RandomHorizontalFlip())
                if random_config["random_rotation"]["use"]: pytorch_list_transforms.append(transforms.RandomRotation(
                    degrees=random_config["random_rotation"]["degrees"]
                ))
                if random_config["random_crop"]["use"]: pytorch_list_transforms.append(transforms.RandomCrop(
                    size=random_config["random_crop"]["size"],
                    pad_if_needed=random_config["random_crop"]["pad_if_needed"],
                    padding_mode=random_config["random_crop"]["padding_mode"]
                ))
                if random_config["random_gaussian_blur"]["use"]: pytorch_list_transforms.append(transforms.GaussianBlur(
                    kernel_size=tuple(random_config["random_gaussian_blur"]["brightness"]),
                    sigma=tuple(random_config["random_gaussian_blur"]["sigma"])
                ))
                if random_config["random_color_jitter"]["use"]: 
                    pytorch_list_transforms.append(transforms.ColorJitter(
                        brightness=random_config["random_color_jitter"]["brightness"],
                        contrast=random_config["random_color_jitter"]["contrast"],
                        saturation=random_config["random_color_jitter"]["saturation"],
                        hue=random_config["random_color_jitter"]["hue"]
                    ))
                if random_config["random_affine"]["use"]: 
                    pytorch_list_transforms.append(transforms.RandomAffine(
                        degrees=random_config["random_affine"]["degrees"],
                        translate=random_config["random_affine"]["translate"],
                        scale=random_config["random_affine"]["scale"],
                        shear=random_config["random_affine"]["shear"],
                        center=random_config["random_affine"]["center"]
                    ))
                if random_config["random_zoom_out"]["use"]:
                    pytorch_list_transforms.append(
                        v2.RandomZoomOut(
                            p=random_config["random_zoom_out"]["probability"]
                        )
                    )
                    
            # TODO: others types of augment here
            #* static transformations for image augemntation - Static Augmentation
            else: 
                pass
        else:
            raise FrameworkError("You choose to augment the dataset, but 'transform_config' is not settled to do augmentation.")
    
    #* standard transforms
    #* resize transformation comes after the random tranformations. Because random transformations could change the images dimensions. The resize transforme ensure
    #* the standard resolution for all images. 
    if transform_config["resize"]["use"] and len(transform_config["resize"]["use"]) == 2: #* transform_config["resize"]["use"] is a list of two numbers (Height, Width).
        # print("Resize Test: {}".format(transform_config["resize"]["use"])) 
        pytorch_list_transforms.append(transforms.Resize(tuple(transform_config["resize"]["use"])))
    
    if transform_config["grayscale"]["use"]:
        pytorch_list_transforms.append(transforms.Grayscale())
    
    if transform_config["totensor"]["use"]:
        pytorch_list_transforms.append(transforms.ToTensor())
    
    if transform_config["normalize_input"]["use"] and not calculate_norm:
        pytorch_list_transforms.append(
            transforms.Normalize(
                mean=tuple(general_config["mean"][transform_config["normalize_input"]["name"]]), 
                std=tuple(general_config["std"][transform_config["normalize_input"]["name"]])
                )
            )
    
    #* transform compose
    transforms_compose = {
        "pytorch_compose": None,
        "albumentation_compose": None
    }
    if len(pytorch_list_transforms) > 0 or len(albumentations_list_transforms) > 0:
        transforms_compose["pytorch_compose"] = transforms.Compose(pytorch_list_transforms)
        transforms_compose["albumentation_compose"] = A.Compose(albumentations_list_transforms)
    else: pass #* In that case, compose continues to be None for Pytorch or Albumentation kind.
        
    return transforms_compose