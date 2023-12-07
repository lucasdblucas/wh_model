import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import math
import os
import numpy as np

from utils.utils_time import get_time_string

def plot_and_save(save_dir, all_distributions):
    
    keys_list = all_distributions.keys()
    ncols = math.ceil(math.sqrt(len(keys_list)))
    nrows = math.ceil(len(keys_list)/ncols)
    # fixado em 1 e len(keys), mas mudar para abarcar todas as possibilidades
    fig, axes = plt.subplots(nrows=1, ncols=len(keys_list), figsize=(20,10))
    index = 0
    
    if 'all' in keys_list:
        if axes.ndim == 1:
            sns.barplot(data = pd.DataFrame.from_dict([all_distributions["all"]]).melt(), x = "variable", y="value", hue="variable", ax=axes[index], dodge=False).set_title('Total Distribution')
            axes[index].tick_params(axis='x', rotation=90)
            axes[index].legend(loc='upper right')
        else:
            sns.barplot(data = pd.DataFrame.from_dict([all_distributions["all"]]).melt(), x = "variable", y="value", hue="variable", ax=axes[index // nrows, index % ncols], dodge=False).set_title('Total Distribution')
            axes[index // nrows, index % ncols].tick_params(axis='x', rotation=90)
            axes[index // nrows, index % ncols].legend(loc='upper right')
        index += 1
    if 'train' in keys_list:
        if axes.ndim == 1:
            sns.barplot(data = pd.DataFrame.from_dict([all_distributions["train"]]).melt(), x = "variable", y="value", hue="variable", ax=axes[index], dodge=False).set_title('Train Distribution')
            axes[index].tick_params(axis='x', rotation=90)
            axes[index].legend(loc='upper right')
        else:    
            sns.barplot(data = pd.DataFrame.from_dict([all_distributions["train"]]).melt(), x = "variable", y="value", hue="variable", ax=axes[index // nrows, index % ncols], dodge=False).set_title('Train Distribution')
            axes[index // nrows, index % ncols].tick_params(axis='x', rotation=90)
            axes[index // nrows, index % ncols].legend(loc='upper right')
        index += 1
    if 'val' in keys_list:
        if axes.ndim == 1:
            sns.barplot(data = pd.DataFrame.from_dict([all_distributions["val"]]).melt(), x = "variable", y="value", hue="variable", ax=axes[index], dodge=False).set_title('Validation Distribution')
            axes[index].tick_params(axis='x', rotation=90)
            axes[index].legend(loc='upper right')
        else:
            sns.barplot(data = pd.DataFrame.from_dict([all_distributions["val"]]).melt(), x = "variable", y="value", hue="variable", ax=axes[index // nrows, index % ncols], dodge=False).set_title('Validation Distribution')
            axes[index // nrows, index % ncols].tick_params(axis='x', rotation=90)
            axes[index // nrows, index % ncols].legend(loc='upper right')
        index += 1
    if 'test' in keys_list:
        if axes.ndim == 1:
            sns.barplot(data = pd.DataFrame.from_dict([all_distributions["test"]]).melt(), x = "variable", y="value", hue="variable", ax=axes[index], dodge=False).set_title('Test Distribution')
            axes[index].tick_params(axis='x', rotation=90)
            axes[index].legend(loc='upper right')
        else:
            sns.barplot(data = pd.DataFrame.from_dict([all_distributions["test"]]).melt(), x = "variable", y="value", hue="variable", ax=axes[index // nrows, index % ncols], dodge=False).set_title('Test Distribution')
            axes[index // nrows, index % ncols].tick_params(axis='x', rotation=90)
            axes[index // nrows, index % ncols].legend(loc='upper right')
        index += 1
    if 'infe' in keys_list:
        if axes.ndim == 1:
            sns.barplot(data = pd.DataFrame.from_dict([all_distributions["infe"]]).melt(), x = "variable", y="value", hue="variable", ax=axes[index], dodge=False).set_title('Inference Distribution')
            axes[index].tick_params(axis='x', rotation=90)
            axes[index].legend(loc='upper right')
        else:
            sns.barplot(data = pd.DataFrame.from_dict([all_distributions["infe"]]).melt(), x = "variable", y="value", hue="variable", ax=axes[index // nrows, index % ncols], dodge=False).set_title('Inference Distribution')
            axes[index // nrows, index % ncols].tick_params(axis='x', rotation=90)
            axes[index // nrows, index % ncols].legend(loc='upper right')
        index += 1
    
    fig.suptitle("Classes Distribution")
    fig_name = get_time_string() + "_distribution_classes.png"
    fig_path = os.path.join(save_dir, fig_name)
    plt.savefig(fig_path)
    plt.close()

def format_report(report, num_classes):
         
    c_report_processed = report.split("\n")
    c_report_processed = [x for x in c_report_processed if x != ""]
    c_report_processed = [x_2 for x in c_report_processed for x_2 in x.split(" ") if x_2 != ""]
    c_report_processed.insert(0, "classes")

    list_of_interest = []
    accuracy = []
    macro_avg = []
    weighted_avg =[]

    limit = (5 * (num_classes + 1))
    for i in range(0, limit, 5):
        sublist = c_report_processed[i: i+5]
        list_of_interest.append(sublist) 

    accuracy.append(c_report_processed[limit])
    limit += 1
    accuracy.extend([" ", " "])
    accuracy.extend(c_report_processed[limit: limit + 2])
    limit += 2
    macro_avg.append(c_report_processed[limit] + "_" + c_report_processed[limit + 1])
    limit += 2
    macro_avg.extend(c_report_processed[limit: limit + 4])
    limit += 4
    weighted_avg.append(c_report_processed[limit] + "_" + c_report_processed[limit + 1])
    limit += 2
    weighted_avg.extend(c_report_processed[limit: ])

    list_of_interest = list_of_interest + [[" ", " ", " ", " ", " "]] + [accuracy] + [macro_avg] + [weighted_avg]
        
    return list_of_interest

def save_matrix(plot_path, plot_title, confusion_matrix, map_ID2Name):
    
    #* dataframe
    confusion_matrix_df = pd.DataFrame(confusion_matrix).rename(columns=map_ID2Name, index=map_ID2Name)
    
    #* fig size
    _, ax = plt.subplots(figsize=(20, 20))
    
    #* font scale size depending on the number of clases
    # len(map_ID2Name.keys()) - 5
    # font_scale = 1 / 
    # sns.set(font_scale=font_scale)
    
    #* heatmap
    #? Verify if this: "cmap=sns.cubehelix_palette(as_cmap=True)" is a better option.
    sns.set(font_scale= 4 / np.sqrt(len(map_ID2Name.keys())))
    sns.heatmap(confusion_matrix_df, annot=True, ax=ax, fmt='d', cmap="crest", linewidth=.5, annot_kws={"size": 35 / np.sqrt(len(map_ID2Name.keys()))}, xticklabels="auto", yticklabels="auto")
    ax.set_title(plot_title, fontdict={'fontsize': 20})
    
    #* save the image
    plt.savefig(plot_path)
    plt.close()

def save_report(plot_path, plot_title, report):
    # save report as image
    fig, ax = plt.subplots(figsize=(8, 8))
    
    font = {
        'family': 'monospace',
        'color':  'darkred',
        'weight': 'normal',
        'size': 12,
    }
    
    ax.text(0, 1, report, va='top', fontdict=font, bbox=dict(facecolor='red', alpha=0.2))
    # Customize the plot
    ax.axis('off')
    ax.set_title(plot_title)
    fig.savefig(plot_path)
    plt.close()
    
def save_loss_error_plot(epochs: list[int] = None, loss: list[float] = None, accuracy: list[float] = None, lr: list[float] = None, plot_path: str = None):
    # # Sample data for loss and accuracy
    # epochs = [1, 2, 3, 4, 5]
    # loss = [0.5, 0.4, 0.3, 0.2, 0.1]
    # accuracy = [0.8, 0.85, 0.9, 0.92, 0.95]

    # Create a figure with subplots
    sns.set(font_scale=1.2)
    plt.figure(figsize=(12, 8))

    # Plot loss
    # plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    sns.lineplot(x=epochs, y=loss, marker="o", label="loss")
    # plt.title('Loss Over Epochs')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')

    # Plot accuracy
    # plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    sns.lineplot(x=epochs, y=accuracy, marker="o", label="accuracy")

    if lr:
        # Plot accuracy
        # plt.subplot(1, 2, 3)  # 1 row, 2 columns, third subplot
        sns.lineplot(x=epochs, y=lr, marker="o", label="Learning Rate")
        # plt.title('Learning Rate Over Epochs')
        # plt.xlabel('Epochs')
        # plt.ylabel('Learning')
    
    plt.title('Metrics over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Values')
    
    # Show the plots
    plt.legend()
    
    #* save the image
    plt.savefig(plot_path)
    plt.close()
    sns.reset_defaults()
    