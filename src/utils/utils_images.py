import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
import tqdm

from collections import defaultdict
 
"""
-> plt.imshow(image, cmap='gray') -> The cmap argument in plt.imshow stands for "colormap". A colormap is a mapping of values in an image to colors. It is used to specify how a grayscale image should be displayed with color. When cmap='gray',
it specifies that the grayscale image should be displayed using a gray colormap, which maps the intensity values of the image to shades of gray. The lighter the intensity value, the lighter the shade of gray in the image, and vice versa. 
Other possible values for cmap include "jet", "viridis", "hot", "cool", and many more. Each of these colormaps maps the image values to a different set of colors, allowing you to create visualizations that highlight different aspects of the data. 
Note that the cmap argument only applies to grayscale images, and has no effect on RGB images. For RGB images, the colors are already specified by the image's channels (red, green, and blue), so no colormap is needed.
"""
# Set the font dictionaries (for plot title and axis titles)
title_font = {
    'fontname':
    'Arial', 
    'size':'14', 
    'color':'black', 
    'weight':'normal',
    'verticalalignment':'bottom'
    }

def display_images(images, labels, channels):
    cols = 5
    rows = len(images) // cols
    
    fig = plt.figure(figsize=(20,25)) #Width, height in inches.
    for index, image in enumerate(images):
        image = image.numpy()
        
        if channels[index] == "Gray_Scale":
            # grayscale image
            image = image.squeeze()
            ax = fig.add_subplot(rows, cols, index+1, aspect="auto") #subplot(nrows, ncols, index, **kwargs)
            ax.imshow(image, cmap='gray')
            
        else:
            # RGB image
            ax = fig.add_subplot(rows, cols, index+1 , aspect="auto")
            ax.imshow(image)
        
        ax.set_title(f"Weight: {round(float(labels[index][0]), 3)}\nHeight: {round(float(labels[index][1]), 3)}\nChannels: {channels[index]}\nResolution: {image.shape}", **title_font)
        # plt.axis('off')
    plt.subplots_adjust(top=1.25) # space between images
    plt.show()


def display_images_from_dataloader(dataloader, num_images):# example usage
    images_for_display = []
    labels_for_display = []
    channels_for_display = []
    stop = 0
        
    for batch, labels in dataloader:
        for index, images in enumerate(batch):
            
            for image in images:
                images_for_display.append(image)
                labels_for_display.append(labels[index])
                channels_for_display.append('RGB' if len(image.shape) == 3 else 'Gray_Scale')
                stop += 1
                
                if stop >= num_images:
                    break
            if stop >= num_images:
                break
        if stop >= num_images:
            break
            
    display_images(images_for_display, labels_for_display, channels_for_display)

def group_info_images(images_folder):
    images_info = {}

    #* _ = subfolders. It is not used
    for folder, _, files in os.walk(images_folder):
        for key, f in enumerate(files):
            image_info = {
                "path": os.path.join(folder, f),
                "name": f[0:-7]  
            }
            
            images_info["image_{}".format(key)] = image_info

    # print("[TEST] num images: {}".format(len(images_info.keys())))
    grouped_info_images = defaultdict(list)
    for _, item in images_info.items():
        grouped_info_images[item["name"]].append(item)
            
    grouped_info_images = dict(grouped_info_images)
    # print("[TEST] num images groups: {}".format(len(grouped_info_images.keys())))

    return grouped_info_images

#* 3h -> 3 images in horizontal orientation
def group_3h_images_and_save(grouped_images_dict, save_path):    
    #* create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    error_msg = "Save path directory doesn't exists."
    assert os.path.exists(save_path), error_msg
    
    len_group = len(grouped_images_dict.keys())
    bar_range = range(len_group)
    with tqdm.tqdm(bar_range, unit="Groups", position=0, dynamic_ncols=True, leave=True) as pbar:
        #* iterate group dict
        for idx_img, (_, group_info) in enumerate(grouped_images_dict.items()):
            pbar.set_description("Group {}/{}".format(idx_img, len_group - 1))
            
            error_msg = "Image Liist less than for elements."
            assert len(group_info) == 4, error_msg
            #sort images. Adding images in a specific sequence
            for image_info in group_info:
                s01 = image_info["path"].split('}')
                s02 = s01[1].split('.')[0] 
                
                #* Is necessary to convert the image file in numpy first, because cv2 will not load a image that has spaces in the string path. And I can't change the name 'Google Drive'.
                if (s02 == "01s") or (s02 == 'SDs'):
                    img_01 = cv2.imdecode(np.fromfile(image_info["path"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                #* not considering this view. Side view, with arms up.
                # if (s02 == "02s") or (s02 == 'SVs'):
                #     img_02 = cv2.imdecode(np.fromfile(image_info["path"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if (s02 == "03s") or (s02 == 'FAs'):
                    img_03 = cv2.imdecode(np.fromfile(image_info["path"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if (s02 == "04s") or (s02 == 'FPs'):
                    img_04 = cv2.imdecode(np.fromfile(image_info["path"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            entry_images = [img_01, img_03, img_04]

            #* check if heights on images are the same. If not, add padding to the image so all the images have the same hight. It allow opencv to concat horizontally all images together
            image_heights = [i.shape[0] for i in entry_images]
            #* set has unique elements, if it has more than one, then all hights is not the same
            if len(set(image_heights)) != 1:
                #* all images will have the hight of the max hight between then
                max_height = max(image_heights)

                for index_02, image in enumerate(entry_images):
                    #* shape = (Hight, Weight)
                    if image.shape[0] != max_height:

                        #* adding pixcels on the top and bottom of the image.
                        dif = max_height - image.shape[0]
                        dif_half = round(dif/2)
                        top, bottom, left, right = dif_half, (dif - dif_half), 0, 0
                        
                        #print(f'\n{entry_images[index_02].shape}, {max_height}')
                        entry_images[index_02] = cv2.copyMakeBorder(entry_images[index_02], top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                        #print(top, bottom, left, right)
                        #print(entry_images[index_02].shape, max_height)

            #* check if the padding went fine
            image_heights = [i.shape[0] for i in entry_images]
            error_msg = "\nPadding has fallen!! Shape img01: {},\nshape img02 : {},\nshape img03 : {}".format(entry_images[0].shape, entry_images[1].shape, entry_images[2].shape)
            assert len(set(image_heights)) == 1, error_msg
            
            #* concatenate images
            concated_image_aux = cv2.hconcat(entry_images)
            
            # print("[TEST] Names: {}, {}".format())
            
            concat_path = os.path.join(save_path, group_info[0]["name"] + '_group_3h.jpg')
            #* encode the image in OpenCV format to one dimension numpy ndarray forma. This is necessary because cv2 will not save in a path that has spaces in the string. And I can't change the name 'Google Drive'.
            is_success, im_buf_arr = cv2.imencode(".jpg", concated_image_aux)
            #* numpy to file and save
            im_buf_arr.tofile(concat_path)
            
            error_msg = "\nSave went wrong. Conversion to numpy went wrong.\nPath: ".format(concat_path)
            assert is_success, error_msg

            pbar.update(1)
            
    return "OK"

#* 2h -> 2 images in horizontal orientation
def group_2h_images_and_save(grouped_images_dict, save_path):    
    #* create directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    error_msg = "Save path directory doesn't exists."
    assert os.path.exists(save_path), error_msg
    
    len_group = len(grouped_images_dict.keys())
    bar_range = range(len_group)
    with tqdm.tqdm(bar_range, unit="Groups", position=0, dynamic_ncols=True, leave=True) as pbar:
        #* iterate group dict
        for idx_img, (_, group_info) in enumerate(grouped_images_dict.items()):
            pbar.set_description("Group {}/{}".format(idx_img, len_group - 1))
            
            error_msg = "Image Liist less than for elements."
            assert len(group_info) == 4, error_msg
            #sort images. Adding images in a specific sequence
            for image_info in group_info:
                s01 = image_info["path"].split('}')
                s02 = s01[1].split('.')[0] 
                
                #* Is necessary to convert the image file in numpy first, because cv2 will not load a image that has spaces in the string path. And I can't change the name 'Google Drive'.
                if (s02 == "01s") or (s02 == 'SDs'):
                    img_01 = cv2.imdecode(np.fromfile(image_info["path"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                #* not considering this view. Side view, with arms up.
                # if (s02 == "02s") or (s02 == 'SVs'):
                #     img_02 = cv2.imdecode(np.fromfile(image_info["path"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if (s02 == "03s") or (s02 == 'FAs'):
                    img_03 = cv2.imdecode(np.fromfile(image_info["path"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                # if (s02 == "04s") or (s02 == 'FPs'):
                #     img_04 = cv2.imdecode(np.fromfile(image_info["path"], dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            entry_images = [img_01, img_03]

            #* check if heights on images are the same. If not, add padding to the image so all the images have the same hight. It allow opencv to concat horizontally all images together
            image_heights = [i.shape[0] for i in entry_images]
            #* set has unique elements, if it has more than one, then all hights is not the same
            if len(set(image_heights)) != 1:
                #* all images will have the hight of the max hight between then
                max_height = max(image_heights)

                for index_02, image in enumerate(entry_images):
                    #* shape = (Hight, Weight)
                    if image.shape[0] != max_height:

                        #* adding pixcels on the top and bottom of the image.
                        dif = max_height - image.shape[0]
                        dif_half = round(dif/2)
                        top, bottom, left, right = dif_half, (dif - dif_half), 0, 0
                        
                        #print(f'\n{entry_images[index_02].shape}, {max_height}')
                        entry_images[index_02] = cv2.copyMakeBorder(entry_images[index_02], top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                        #print(top, bottom, left, right)
                        #print(entry_images[index_02].shape, max_height)

            #* check if the padding went fine
            image_heights = [i.shape[0] for i in entry_images]
            error_msg = "\nPadding has fallen!! Shape img01: {},\nshape img03 : {}".format(entry_images[0].shape, entry_images[1].shape)
            assert len(set(image_heights)) == 1, error_msg
            
            #* concatenate images
            concated_image_aux = cv2.hconcat(entry_images)
            
            # print("[TEST] Names: {}, {}".format())
            
            concat_path = os.path.join(save_path, group_info[0]["name"] + '_group_2h.jpg')
            #* encode the image in OpenCV format to one dimension numpy ndarray forma. This is necessary because cv2 will not save in a path that has spaces in the string. And I can't change the name 'Google Drive'.
            is_success, im_buf_arr = cv2.imencode(".jpg", concated_image_aux)
            #* numpy to file and save
            im_buf_arr.tofile(concat_path)
            
            error_msg = "\nSave went wrong. Conversion to numpy went wrong.\nPath: ".format(concat_path)
            assert is_success, error_msg

            pbar.update(1)
            
    return "OK"