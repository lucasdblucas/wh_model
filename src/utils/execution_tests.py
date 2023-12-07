import os

from utils_images import group_info_images, group_3h_images_and_save, group_2h_images_and_save
from beeprint import pp

images_folder = "C:/Users/lucas/GoogleDrive/Meu Drive/Colab Notebooks/Segmentação/segmented_images/biotonix/separate_images"

grouped_info_images_dict = group_info_images(images_folder=images_folder)

#* checkin if there area exacly 4 elements for each item
str_error = "There is a group that have not 4 elements"
for ids, (k, item) in enumerate(grouped_info_images_dict.items()):
    assert len(item) == 4, str_error

parent_path = os.path.dirname(images_folder)
save_path = os.path.join(parent_path, "grouped_images_2h")

#* 
response = group_2h_images_and_save(
    grouped_images_dict=grouped_info_images_dict,
    save_path=save_path
)

print("[FINAL]: {}".format(response))