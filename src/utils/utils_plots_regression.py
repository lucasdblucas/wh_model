import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import numpy as np

from sklearn.metrics import PredictionErrorDisplay, median_absolute_error, r2_score, mean_squared_error, mean_absolute_error
from utils.utils_time import get_time_string

def plot_and_save(save_dir, targets_over_sets):
    
    keys_list = targets_over_sets.keys()
    size_key_list = len(keys_list) * targets_over_sets["all"].shape[1] #* two plots per key (weight, height)
    ncols = math.ceil(math.sqrt(size_key_list))
    nrows = math.ceil(size_key_list/ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,10))
    index = 0
    
    aux = max(nrows, ncols)
    if 'all' in keys_list:
        if axes.ndim == 1:
            axes[index].hist(targets_over_sets["all"][:,0], bins=100, density=True)
            axes[index].set_ylabel("Probability")
            axes[index].set_xlabel("Weight(Kg)")
            axes[index].set_xlim(7.95445, 182.95455)
            # axes[index].set_ylim(0.0, 0.030346402982524046)
            axes[index].set_title("Weight Distribution - Total")
            index += 1
            axes[index].hist(targets_over_sets["all"][:,1], bins=100, density=True)
            axes[index].set_ylabel("Probability")
            axes[index].set_xlabel("Height(cm)")
            axes[index].set_xlim(88.51900000000002, 208.661)
            # axes[index].set_ylim(0.0, 0.08640552330868853)
            axes[index].set_title("Height Distribution - Total")
            index += 1
            axes[index].hist(targets_over_sets["all"][:,2], bins=100, density=True)
            axes[index].set_ylabel("Probability")
            axes[index].set_xlabel("BMI")
            # axes[index].set_xlim()
            # axes[index].set_ylim()
            axes[index].set_title("BMI Distribution - Total")
            index += 1
        else:
            index_aux = index // aux, index % aux
            axes[index_aux].hist(targets_over_sets["all"][:,0], bins=100, density=True)
            axes[index_aux].set_ylabel("Probability")
            axes[index_aux].set_xlabel("Weight(Kg)")
            axes[index_aux].set_xlim(7.95445, 182.95455)
            # axes[index_aux].set_ylim(0.0, 0.030346402982524046)
            axes[index_aux].set_title("Weight Distribution - Total")
            index += 1
            index_aux = index // aux, index % aux
            axes[index_aux].hist(targets_over_sets["all"][:,1], bins=100, density=True)
            axes[index_aux].set_ylabel("Probability")
            axes[index_aux].set_xlabel("Height(cm)")
            axes[index_aux].set_xlim(88.51900000000002, 208.661)
            # axes[index_aux].set_ylim(0.0, 0.08640552330868853)
            axes[index_aux].set_title("Height Distribution - Total")
            index += 1
            index_aux = index // aux, index % aux
            axes[index_aux].hist(targets_over_sets["all"][:,2], bins=100, density=True)
            axes[index_aux].set_ylabel("Probability")
            axes[index_aux].set_xlabel("BMI")
            # axes[index_aux].set_xlim()
            # axes[index_aux].set_ylim()
            axes[index_aux].set_title("BMI Distribution - Total")
            index += 1
            
    if 'train' in keys_list:
        if axes.ndim == 1:
            axes[index].hist(targets_over_sets["train"][:,0], bins=100, density=True)
            axes[index].set_ylabel("Probability")
            axes[index].set_xlabel("Weight(Kg)")
            axes[index].set_xlim(7.95445, 182.95455)
            # axes[index].set_ylim(0.0, 0.030346402982524046)
            axes[index].set_title("Weight Distribution - Train")
            index += 1
            axes[index].hist(targets_over_sets["train"][:,1], bins=100, density=True)
            axes[index].set_ylabel("Probability")
            axes[index].set_xlabel("Height(cm)")
            axes[index].set_xlim(88.51900000000002, 208.661)
            # axes[index].set_ylim(0.0, 0.08640552330868853)
            axes[index].set_title("Height Distribution - Train")
            index += 1
            axes[index].hist(targets_over_sets["all"][:,2], bins=100, density=True)
            axes[index].set_ylabel("Probability")
            axes[index].set_xlabel("BMI")
            # axes[index].set_xlim()
            # axes[index].set_ylim()
            axes[index].set_title("BMI Distribution - Total")
            index += 1
        else:    
            index_aux = index // aux, index % aux
            axes[index_aux].hist(targets_over_sets["train"][:,0], bins=100, density=True)
            axes[index_aux].set_ylabel("Probability")
            axes[index_aux].set_xlabel("Weight(Kg)")
            axes[index_aux].set_xlim(7.95445, 182.95455)
            # axes[index_aux].set_ylim(0.0, 0.030346402982524046)
            axes[index_aux].set_title("Weight Distribution - Train")
            index += 1
            index_aux = index // aux, index % aux
            axes[index_aux].hist(targets_over_sets["train"][:,1], bins=100, density=True)
            axes[index_aux].set_ylabel("Probability")
            axes[index_aux].set_xlabel("Height(cm)")
            axes[index_aux].set_xlim(88.51900000000002, 208.661)
            # axes[index_aux].set_ylim(0.0, 0.08640552330868853)
            axes[index_aux].set_title("Height Distribution - Train")
            index += 1
            index_aux = index // aux, index % aux
            axes[index_aux].hist(targets_over_sets["all"][:,2], bins=100, density=True)
            axes[index_aux].set_ylabel("Probability")
            axes[index_aux].set_xlabel("BMI")
            # axes[index_aux].set_xlim()
            # axes[index_aux].set_ylim()
            axes[index_aux].set_title("BMI Distribution - Total")
            index += 1
            
    if 'val' in keys_list:
        if axes.ndim == 1:
            axes[index].hist(targets_over_sets["val"][:,0], bins=100, density=True)
            axes[index].set_ylabel("Probability")
            axes[index].set_xlabel("Weight(Kg)")
            axes[index].set_xlim(7.95445, 182.95455)
            # axes[index].set_ylim(0.0, 0.030346402982524046)
            axes[index].set_title("Weight Distribution - Validation")
            index += 1
            axes[index].hist(targets_over_sets["val"][:,1], bins=100, density=True)
            axes[index].set_ylabel("Probability")
            axes[index].set_xlabel("Height(cm)")
            axes[index].set_xlim(88.51900000000002, 208.661)
            # axes[index].set_ylim(0.0, 0.08640552330868853)
            axes[index].set_title("Height Distribution - Validation")
            index += 1
            axes[index].hist(targets_over_sets["all"][:,2], bins=100, density=True)
            axes[index].set_ylabel("Probability")
            axes[index].set_xlabel("BMI")
            # axes[index].set_xlim()
            # axes[index].set_ylim()
            axes[index].set_title("BMI Distribution - Total")
            index += 1
        else:
            index_aux = index // aux, index % aux
            axes[index_aux].hist(targets_over_sets["val"][:,0], bins=100, density=True)
            axes[index_aux].set_ylabel("Probability")
            axes[index_aux].set_xlabel("Weight(Kg)")
            axes[index_aux].set_xlim(7.95445, 182.95455)
            # axes[index_aux].set_ylim(0.0, 0.030346402982524046)
            axes[index_aux].set_title("Weight Distribution - Validation")
            index += 1
            index_aux = index // aux, index % aux
            axes[index_aux].hist(targets_over_sets["val"][:,1], bins=100, density=True)
            axes[index_aux].set_ylabel("Probability")
            axes[index_aux].set_xlabel("Height(cm)")
            axes[index_aux].set_xlim(88.51900000000002, 208.661)
            # axes[index_aux].set_ylim(0.0, 0.08640552330868853)
            axes[index_aux].set_title("Height Distribution - Validation")
            index += 1
            index_aux = index // aux, index % aux
            axes[index_aux].hist(targets_over_sets["all"][:,2], bins=100, density=True)
            axes[index_aux].set_ylabel("Probability")
            axes[index_aux].set_xlabel("BMI")
            # axes[index_aux].set_xlim()
            # axes[index_aux].set_ylim()
            axes[index_aux].set_title("BMI Distribution - Total")
            index += 1
            
    if 'test' in keys_list:
        if axes.ndim == 1:
            axes[index].hist(targets_over_sets["test"][:,0], bins=100, density=True)
            axes[index].set_ylabel("Probability")
            axes[index].set_xlabel("Weight(Kg)")
            axes[index].set_xlim(7.95445, 182.95455)
            # axes[index].set_ylim(0.0, 0.030346402982524046)
            axes[index].set_title("Weight Distribution - Test")
            index += 1
            axes[index].hist(targets_over_sets["test"][:,1], bins=100, density=True)
            axes[index].set_ylabel("Probability")
            axes[index].set_xlabel("Height(cm)")
            axes[index].set_xlim(88.51900000000002, 208.661)
            # axes[index].set_ylim(0.0, 0.08640552330868853)
            axes[index].set_title("Height Distribution - Test")
            index += 1
            axes[index].hist(targets_over_sets["all"][:,2], bins=100, density=True)
            axes[index].set_ylabel("Probability")
            axes[index].set_xlabel("BMI")
            # axes[index].set_xlim()
            # axes[index].set_ylim()
            axes[index].set_title("BMI Distribution - Total")
            index += 1
        else:
            index_aux = index // aux, index % aux
            axes[index_aux].hist(targets_over_sets["test"][:,0], bins=100, density=True)
            axes[index_aux].set_ylabel("Probability")
            axes[index_aux].set_xlabel("Weight(Kg)")
            axes[index_aux].set_xlim(7.95445, 182.95455)
            # axes[index_aux].set_ylim(0.0, 0.030346402982524046)
            axes[index_aux].set_title("Weight Distribution - Test")
            index += 1
            index_aux = index // aux, index % aux
            axes[index_aux].hist(targets_over_sets["test"][:,1], bins=100, density=True)
            axes[index_aux].set_ylabel("Probability")
            axes[index_aux].set_xlabel("Height(cm)")
            axes[index_aux].set_xlim(88.51900000000002, 208.661)
            # axes[index_aux].set_ylim(0.0, 0.08640552330868853)
            axes[index_aux].set_title("Height Distribution - Test")
            index += 1
            index_aux = index // aux, index % aux
            axes[index_aux].hist(targets_over_sets["all"][:,2], bins=100, density=True)
            axes[index_aux].set_ylabel("Probability")
            axes[index_aux].set_xlabel("BMI")
            # axes[index_aux].set_xlim()
            # axes[index_aux].set_ylim()
            axes[index_aux].set_title("BMI Distribution - Total")
            index += 1
            
    if 'infe' in keys_list:
        if axes.ndim == 1:
            axes[index].hist(targets_over_sets["infe"][:,0], bins=100, density=True)
            axes[index].set_ylabel("Probability")
            axes[index].set_xlabel("Weight(Kg)")
            axes[index].set_xlim(7.95445, 182.95455)
            # axes[index].set_ylim(0.0, 0.030346402982524046)
            axes[index].set_title("Weight Distribution - Inference")
            index += 1
            axes[index].hist(targets_over_sets["infe"][:,1], bins=100, density=True)
            axes[index].set_ylabel("Probability")
            axes[index].set_xlabel("Height(cm)")
            axes[index].set_xlim(88.51900000000002, 208.661)
            # axes[index].set_ylim(0.0, 0.08640552330868853)
            axes[index].set_title("Height Distribution - Inference")
            index += 1
            axes[index].hist(targets_over_sets["all"][:,2], bins=100, density=True)
            axes[index].set_ylabel("Probability")
            axes[index].set_xlabel("BMI")
            # axes[index].set_xlim()
            # axes[index].set_ylim()
            axes[index].set_title("BMI Distribution - Total")
            index += 1
        else:
            index_aux = index // aux, index % aux
            axes[index_aux].hist(targets_over_sets["infe"][:,0], bins=100, density=True)
            axes[index_aux].set_ylabel("Probability")
            axes[index_aux].set_xlabel("Weight(Kg)")
            axes[index_aux].set_xlim(7.95445, 182.95455)
            axes[index_aux].set_ylim(0.0, 0.030346402982524046)
            axes[index_aux].set_title("Weight Distribution - Inference")
            index += 1
            index_aux = index // aux, index % aux
            axes[index_aux].hist(targets_over_sets["infe"][:,1], bins=100, density=True)
            axes[index_aux].set_ylabel("Probability")
            axes[index_aux].set_xlabel("Height(cm)")
            axes[index_aux].set_xlim(88.51900000000002, 208.661)
            # axes[index_aux].set_ylim(0.0, 0.08640552330868853)
            axes[index_aux].set_title("Height Distribution - Inference")
            index += 1
            index_aux = index // aux, index % aux
            axes[index_aux].hist(targets_over_sets["all"][:,2], bins=100, density=True)
            axes[index_aux].set_ylabel("Probability")
            axes[index_aux].set_xlabel("BMI")
            # axes[index_aux].set_xlim()
            # axes[index_aux].set_ylim()
            axes[index_aux].set_title("BMI Distribution - Total")
            index += 1
    
    fig.suptitle("Ditribution over Dataset", y=1.05)
    plt.tight_layout()
    
    fig_name = get_time_string() + "_distribution.png"
    fig_path = os.path.join(save_dir, fig_name)
    plt.savefig(fig_path)
    plt.close()

def plot_and_save_sns_countplot(save_dir, targets_over_sets):
    
    keys_list = targets_over_sets.keys()
    size_key_list = len(keys_list) * targets_over_sets["all"].shape[1] #* two plots per key (weight, height)
    ncols = math.ceil(math.sqrt(size_key_list))
    nrows = math.ceil(size_key_list/ncols)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(20,10))
    index = 0
    
    aux = max(nrows, ncols)
    if 'all' in keys_list:
        if axes.ndim == 1:
            
            sns.histplot(data=targets_over_sets["all"][:,0], bins=50, ax=axes[index])
            axes[index].set_ylabel("Count")
            axes[index].set_xlabel("Weight(Kg)")
            axes[index].set_xlim(7.95445, 182.95455)
            axes[index].set_title("Weight Distribution - Total")
            
            index += 1
            
            sns.histplot(targets_over_sets["all"][:,1], bins=170, ax=axes[index])
            axes[index].set_ylabel("Count")
            axes[index].set_xlabel("Height(cm)")
            axes[index].set_xlim(88.51900000000002, 208.661)
            axes[index].set_title("Height Distribution - Total")
            
            index += 1
            
            sns.histplot(targets_over_sets["all"][:,2], bins=50, ax=axes[index])
            axes[index].set_ylabel("Count")
            axes[index].set_xlabel("BMI")
            # axes[index].set_xlim()
            # axes[index].set_ylim()
            axes[index].set_title("BMI Distribution - Total")
            index += 1
        else:
            index_aux = index // aux, index % aux
            sns.histplot(targets_over_sets["all"][:,0], bins=50, ax=axes[index_aux])
            axes[index_aux].set_ylabel("Count")
            axes[index_aux].set_xlabel("Weight(Kg)")
            axes[index_aux].set_xlim(7.95445, 182.95455)
            axes[index_aux].set_title("Weight Distribution - Total")
            
            index += 1
            index_aux = index // aux, index % aux
            
            sns.histplot(targets_over_sets["all"][:,1], bins=170, ax=axes[index_aux])
            axes[index_aux].set_ylabel("Count")
            axes[index_aux].set_xlabel("Height(cm)")
            axes[index_aux].set_xlim(88.51900000000002, 208.661)
            axes[index_aux].set_title("Height Distribution - Total")
            
            index += 1
            index_aux = index // aux, index % aux
            
            sns.histplot(targets_over_sets["all"][:,2], bins=50, ax=axes[index_aux])
            axes[index_aux].set_ylabel("Count")
            axes[index_aux].set_xlabel("BMI")
            # axes[index_aux].set_xlim()
            # axes[index_aux].set_ylim()
            axes[index_aux].set_title("BMI Distribution - Total")
            index += 1
            
    if 'train' in keys_list:
        if axes.ndim == 1:
            sns.histplot(targets_over_sets["train"][:,0], bins=50, ax=axes[index])
            axes[index].set_ylabel("Count")
            axes[index].set_xlabel("Weight(Kg)")
            axes[index].set_xlim(7.95445, 182.95455)
            axes[index].set_title("Weight Distribution - Train")
            
            index += 1
            
            sns.histplot(targets_over_sets["train"][:,1], bins=170, ax=axes[index])
            axes[index].set_ylabel("Count")
            axes[index].set_xlabel("Height(cm)")
            axes[index].set_xlim(88.51900000000002, 208.661)
            axes[index].set_title("Height Distribution - Train")
            
            index += 1
            
            sns.histplot(targets_over_sets["train"][:,2], bins=50, ax=axes[index])
            axes[index].set_ylabel("Count")
            axes[index].set_xlabel("BMI")
            # axes[index].set_xlim()
            # axes[index].set_ylim()
            axes[index].set_title("BMI Distribution - Total")
            
            index += 1
        else:    
            index_aux = index // aux, index % aux
            
            sns.histplot(targets_over_sets["train"][:,0], bins=50, ax=axes[index_aux])
            axes[index_aux].set_ylabel("Count")
            axes[index_aux].set_xlabel("Weight(Kg)")
            axes[index_aux].set_xlim(7.95445, 182.95455)
            axes[index_aux].set_title("Weight Distribution - Train")
            
            index += 1
            index_aux = index // aux, index % aux
            
            sns.histplot(targets_over_sets["train"][:,1], bins=170, ax=axes[index_aux])
            axes[index_aux].set_ylabel("Count")
            axes[index_aux].set_xlabel("Height(cm)")
            axes[index_aux].set_xlim(88.51900000000002, 208.661)
            axes[index_aux].set_title("Height Distribution - Train")
            
            index += 1
            index_aux = index // aux, index % aux
            
            sns.histplot(targets_over_sets["train"][:,2], bins=50, ax=axes[index_aux])
            axes[index_aux].set_ylabel("Count")
            axes[index_aux].set_xlabel("BMI")
            # axes[index_aux].set_xlim()
            # axes[index_aux].set_ylim()
            axes[index_aux].set_title("BMI Distribution - Total")
            
            index += 1
            
    if 'val' in keys_list:
        # if axes.ndim == 1:
        #     axes[index].hist(targets_over_sets["val"][:,0], bins=100, density=True)
        #     axes[index].set_ylabel("Probability")
        #     axes[index].set_xlabel("Weight(Kg)")
        #     axes[index].set_xlim(7.95445, 182.95455)
        #     # axes[index].set_ylim(0.0, 0.030346402982524046)
        #     axes[index].set_title("Weight Distribution - Validation")
        #     index += 1
        #     axes[index].hist(targets_over_sets["val"][:,1], bins=100, density=True)
        #     axes[index].set_ylabel("Probability")
        #     axes[index].set_xlabel("Height(cm)")
        #     axes[index].set_xlim(88.51900000000002, 208.661)
        #     # axes[index].set_ylim(0.0, 0.08640552330868853)
        #     axes[index].set_title("Height Distribution - Validation")
        #     index += 1
        #     axes[index].hist(targets_over_sets["all"][:,2], bins=100, density=True)
        #     axes[index].set_ylabel("Probability")
        #     axes[index].set_xlabel("BMI")
        #     # axes[index].set_xlim()
        #     # axes[index].set_ylim()
        #     axes[index].set_title("BMI Distribution - Total")
        #     index += 1
        # else:
        #     index_aux = index // aux, index % aux
        #     axes[index_aux].hist(targets_over_sets["val"][:,0], bins=100, density=True)
        #     axes[index_aux].set_ylabel("Probability")
        #     axes[index_aux].set_xlabel("Weight(Kg)")
        #     axes[index_aux].set_xlim(7.95445, 182.95455)
        #     # axes[index_aux].set_ylim(0.0, 0.030346402982524046)
        #     axes[index_aux].set_title("Weight Distribution - Validation")
        #     index += 1
        #     index_aux = index // aux, index % aux
        #     axes[index_aux].hist(targets_over_sets["val"][:,1], bins=100, density=True)
        #     axes[index_aux].set_ylabel("Probability")
        #     axes[index_aux].set_xlabel("Height(cm)")
        #     axes[index_aux].set_xlim(88.51900000000002, 208.661)
        #     # axes[index_aux].set_ylim(0.0, 0.08640552330868853)
        #     axes[index_aux].set_title("Height Distribution - Validation")
        #     index += 1
        #     index_aux = index // aux, index % aux
        #     axes[index_aux].hist(targets_over_sets["all"][:,2], bins=100, density=True)
        #     axes[index_aux].set_ylabel("Probability")
        #     axes[index_aux].set_xlabel("BMI")
        #     # axes[index_aux].set_xlim()
        #     # axes[index_aux].set_ylim()
        #     axes[index_aux].set_title("BMI Distribution - Total")
        #     index += 1
        pass
            
    if 'test' in keys_list:
        if axes.ndim == 1:
            sns.histplot(targets_over_sets["test"][:,0], bins=50, ax=axes[index])
            axes[index].set_ylabel("Count")
            axes[index].set_xlabel("Weight(Kg)")
            axes[index].set_xlim(7.95445, 182.95455)
            axes[index].set_title("Weight Distribution - Test")
            
            index += 1
            
            sns.histplot(targets_over_sets["test"][:,1], bins=45, ax=axes[index])
            axes[index].set_ylabel("Count")
            axes[index].set_xlabel("Height(cm)")
            axes[index].set_xlim(88.51900000000002, 208.661)
            axes[index].set_title("Height Distribution - Test")
            
            index += 1
            
            sns.histplot(targets_over_sets["test"][:,2], bins=50, ax=axes[index])
            axes[index].set_ylabel("Count")
            axes[index].set_xlabel("BMI")
            # axes[index].set_xlim()
            # axes[index].set_ylim()
            axes[index].set_title("BMI Distribution - Total")
            
            index += 1
        else:
            index_aux = index // aux, index % aux
            
            sns.histplot(targets_over_sets["test"][:,0], bins=50, ax=axes[index_aux])
            axes[index_aux].set_ylabel("Count")
            axes[index_aux].set_xlabel("Weight(Kg)")
            axes[index_aux].set_xlim(7.95445, 182.95455)
            axes[index_aux].set_title("Weight Distribution - Test")
            
            index += 1
            index_aux = index // aux, index % aux
            
            sns.histplot(targets_over_sets["test"][:,1], bins=45, ax=axes[index_aux])
            axes[index_aux].set_ylabel("Count")
            axes[index_aux].set_xlabel("Height(cm)")
            axes[index_aux].set_xlim(88.51900000000002, 208.661)
            axes[index_aux].set_title("Height Distribution - Test")
            
            index += 1
            index_aux = index // aux, index % aux
            
            sns.histplot(targets_over_sets["test"][:,2], bins=50, ax=axes[index_aux])
            axes[index_aux].set_ylabel("Count")
            axes[index_aux].set_xlabel("BMI")
            # axes[index_aux].set_xlim()
            # axes[index_aux].set_ylim()
            axes[index_aux].set_title("BMI Distribution - Total")
            
            index += 1
            
    if 'infe' in keys_list:
        # if axes.ndim == 1:
        #     axes[index].hist(targets_over_sets["infe"][:,0], bins=100, density=True)
        #     axes[index].set_ylabel("Probability")
        #     axes[index].set_xlabel("Weight(Kg)")
        #     axes[index].set_xlim(7.95445, 182.95455)
        #     # axes[index].set_ylim(0.0, 0.030346402982524046)
        #     axes[index].set_title("Weight Distribution - Inference")
        #     index += 1
        #     axes[index].hist(targets_over_sets["infe"][:,1], bins=100, density=True)
        #     axes[index].set_ylabel("Probability")
        #     axes[index].set_xlabel("Height(cm)")
        #     axes[index].set_xlim(88.51900000000002, 208.661)
        #     # axes[index].set_ylim(0.0, 0.08640552330868853)
        #     axes[index].set_title("Height Distribution - Inference")
        #     index += 1
        #     axes[index].hist(targets_over_sets["all"][:,2], bins=100, density=True)
        #     axes[index].set_ylabel("Probability")
        #     axes[index].set_xlabel("BMI")
        #     # axes[index].set_xlim()
        #     # axes[index].set_ylim()
        #     axes[index].set_title("BMI Distribution - Total")
        #     index += 1
        # else:
        #     index_aux = index // aux, index % aux
        #     axes[index_aux].hist(targets_over_sets["infe"][:,0], bins=100, density=True)
        #     axes[index_aux].set_ylabel("Probability")
        #     axes[index_aux].set_xlabel("Weight(Kg)")
        #     axes[index_aux].set_xlim(7.95445, 182.95455)
        #     axes[index_aux].set_ylim(0.0, 0.030346402982524046)
        #     axes[index_aux].set_title("Weight Distribution - Inference")
        #     index += 1
        #     index_aux = index // aux, index % aux
        #     axes[index_aux].hist(targets_over_sets["infe"][:,1], bins=100, density=True)
        #     axes[index_aux].set_ylabel("Probability")
        #     axes[index_aux].set_xlabel("Height(cm)")
        #     axes[index_aux].set_xlim(88.51900000000002, 208.661)
        #     # axes[index_aux].set_ylim(0.0, 0.08640552330868853)
        #     axes[index_aux].set_title("Height Distribution - Inference")
        #     index += 1
        #     index_aux = index // aux, index % aux
        #     axes[index_aux].hist(targets_over_sets["all"][:,2], bins=100, density=True)
        #     axes[index_aux].set_ylabel("Probability")
        #     axes[index_aux].set_xlabel("BMI")
        #     # axes[index_aux].set_xlim()
        #     # axes[index_aux].set_ylim()
        #     axes[index_aux].set_title("BMI Distribution - Total")
        #     index += 1
        pass
    
    fig.suptitle("Ditribution over Dataset", y=1.05)
    plt.tight_layout()
    
    fig_name = get_time_string() + "_distribution.png"
    fig_path = os.path.join(save_dir, fig_name)
    plt.savefig(fig_path)
    plt.close()

def compute_score(y_true, y_pred):
    return {
        "R2": f"{r2_score(y_true, y_pred):.3f}",
        "MAE": f"{mean_absolute_error(y_true, y_pred):.3f}",
        "MedAE": f"{median_absolute_error(y_true, y_pred):.3f}",
        "RMSE": f"{np.sqrt(mean_squared_error(y_true, y_pred)):.3f}"
    }

def save_plot_prediction(fig_path, plot_title, y_true_wh, y_pred_wh, ids):
    fig, axes = plt.subplots(2, 2, figsize=(13, 13))
    
    y_true_w = np.array([y[0] for y in y_true_wh])
    y_true_h = np.array([y[1] for y in y_true_wh])
    y_pred_w = np.array([y[0] for y in y_pred_wh])
    y_pred_h = np.array([y[1] for y in y_pred_wh])
    
    display_01 = PredictionErrorDisplay.from_predictions(
        y_true=y_true_w,
        y_pred=y_pred_w,
        kind="actual_vs_predicted",
        ax=axes[0, 0],
        scatter_kwargs={"alpha": 0.5}
    )
    display_02 = PredictionErrorDisplay.from_predictions(
        y_true=y_true_w,
        y_pred=y_pred_w,
        kind="residual_vs_predicted",
        ax=axes[0, 1],
        scatter_kwargs={"alpha": 0.5}
    )
    
    #* top 10 errors
    n = 10 if len(y_true_w) >= 10 else len(y_true_w)
    
    np_diff_w = np.sqrt(((y_true_w - y_pred_w) **2))
    np_diff_h = np.sqrt(((y_true_h - y_pred_h) **2))
    
    index_w = np.argpartition(np_diff_w, -n)[-n:]
    index_h = np.argpartition(np_diff_h, -n)[-n:]
    
    for idx in index_w:
        axes[0, 0].annotate(f"id:{ids[idx]}", xy=(display_01.y_pred[idx], display_01.y_true[idx]), fontsize=10, color='g')
        axes[0, 1].annotate(f"id:{ids[idx]}", xy=(display_02.y_pred[idx], display_02.y_true[idx] - display_02.y_pred[idx]), fontsize=10, color='g')
        axes[0, 0].scatter(display_01.y_pred[idx], display_01.y_true[idx], color="r")
        axes[0, 1].scatter(display_02.y_pred[idx], display_02.y_true[idx] - display_02.y_pred[idx], color="r")
        
    
    display_03 = PredictionErrorDisplay.from_predictions(
        y_true=y_true_h,
        y_pred=y_pred_h,
        kind="actual_vs_predicted",
        ax=axes[1, 0],
        scatter_kwargs={"alpha": 0.5}
    )
    display_04 = PredictionErrorDisplay.from_predictions(
        y_true=y_true_h,
        y_pred=y_pred_h,
        kind="residual_vs_predicted",
        ax=axes[1, 1],
        scatter_kwargs={"alpha": 0.5}
    )
    
    for idx in index_h:
        axes[1, 0].annotate(f"id:{ids[idx]}", xy=(display_03.y_pred[idx], display_03.y_true[idx]), fontsize=10, color='g')
        axes[1, 1].annotate(f"id:{ids[idx]}", xy=(display_04.y_pred[idx], display_04.y_true[idx] - display_04.y_pred[idx]), fontsize=10, color='g')
        axes[1, 0].scatter(display_03.y_pred[idx], display_03.y_true[idx], color="r")
        axes[1, 1].scatter(display_04.y_pred[idx], display_04.y_true[idx] - display_04.y_pred[idx], color="r")
    
    axes[0, 0].set_title("Weight - Actual x Predicted")
    axes[0, 1].set_title("Weight - Residual x Predicted")
    axes[1, 0].set_title("Height - Actual x Predicted")
    axes[1, 1].set_title("Height - Residual x Predicted")
    
    weight_score = compute_score(y_true=y_true_w, y_pred=y_pred_w)
    height_score = compute_score(y_true=y_true_h, y_pred=y_pred_h)
    
    #* weight actual_vs_predicted
    for key in weight_score.keys():
        axes[0, 0].plot([], [], " ", label="{}={}".format(key, weight_score[key]))
            
    axes[0, 0].tick_params(axis='x', rotation=90)
    axes[0, 0].legend(loc="upper left")
    
    #* weight residual_vs_predicted
    axes[0, 1].tick_params(axis='x', rotation=90)
    
    #* height actual_vs_predicted
    for key in height_score.keys():
        axes[1, 0].plot([], [], " ", label="{}={}".format(key, height_score[key]))
            
    axes[1, 0].legend(loc="upper left")
    axes[1, 0].tick_params(axis='x', rotation=90)
    
    #* height residual_vs_predicted
    axes[1, 1].tick_params(axis='x', rotation=90)
    
    fig.suptitle(plot_title)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()
    
def save_plot_prediction_1out(fig_path, plot_title, y_true, y_pred, ids):
    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    display_01 = PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        ax=axes[0],
        scatter_kwargs={"alpha": 0.5}
    )
    display_02 = PredictionErrorDisplay.from_predictions(
        y_true=y_true,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        ax=axes[1],
        scatter_kwargs={"alpha": 0.5}
    )
    
    #* top 10 errors
    n = 10 if len(y_true) >= 10 else len(y_true)
    
    np_diff = np.sqrt(((y_true - y_pred) **2))
    
    index = np.argpartition(np_diff, -n)[-n:]
    
    for idx in index:
        axes[0].annotate(f"id:{ids[idx]}", xy=(display_01.y_pred[idx], display_01.y_true[idx]), fontsize=10, color='g')
        axes[1].annotate(f"id:{ids[idx]}", xy=(display_02.y_pred[idx], display_02.y_true[idx] - display_02.y_pred[idx]), fontsize=10, color='g')
        axes[0].scatter(display_01.y_pred[idx], display_01.y_true[idx], color="r")
        axes[1].scatter(display_02.y_pred[idx], display_02.y_true[idx] - display_02.y_pred[idx], color="r")
    
    axes[0].set_title("Actual x Predicted")
    axes[1].set_title("Residual x Predicted")
    
    score = compute_score(y_true=y_true, y_pred=y_pred)
    
    #* actual_vs_predicted
    for key in score.keys():
        axes[0].plot([], [], " ", label="{}={}".format(key, score[key]))
            
    axes[0].tick_params(axis='x', rotation=90)
    axes[0].legend(loc="upper left")
    
    #* residual_vs_predicted
    axes[1].tick_params(axis='x', rotation=90)
    
    fig.suptitle(plot_title)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()