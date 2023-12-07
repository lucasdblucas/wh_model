import torch
import numpy as np
import tqdm
import time
import torch.nn.functional as F

from statistics import mean
from utils.utils_time import convert_seconds
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# define a function to calculate the metrics
# In scikit-learn's classification_report, the "support" refers to the number of samples in each class that were used to evaluate the performance of the classifier.
def calculate_metrics(y_true, y_pred, classes, labels):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    classification_report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0, target_names=classes, labels=labels)
    classification_report_str = classification_report(y_true, y_pred, zero_division=0, target_names=classes, labels=labels)
    return accuracy, precision, recall, f1, [classification_report_dict, classification_report_str], confusion_matrix(y_true, y_pred)

def train_model(model, optimizer, loader, criterion, epoch_info, start_time, device=torch.device('cuda')):
    model.train()
    
    y_pred_accumulator = []
    y_target_accumulator = []
    
    running_loss = []
    running_accuracy = []
    
    running_time = 0.0
    
    #* How many times the element had a incorrectly prediction
    worst_elements = {"key_"+str(x):{"loss": 0.0, "path": "", "id": ""} for x in range(10)}
    
    #* classes and names
    name_to_class_map = loader.dataset.name_to_class_map
    classes = [k for k, v in name_to_class_map.items()]
    labels = [v for k, v in name_to_class_map.items()]
    
    loader_size = len(loader)
    bar_range = range(loader_size)
    with tqdm.tqdm(bar_range, unit="batchs", position=1, dynamic_ncols=True) as bar:
        for batch_idx, (data, info, metadata, target) in enumerate(loader):
            #//TODO: veririficar se info pode ser retirado daqui, o acesso as outras variáveis são possíveis atraves de loader.dataset.""""
            #* bar
            bar.set_description("Train Epoch-{}/{} Batch-{}/{}".format(epoch_info[0], epoch_info[1] - 1, batch_idx, loader_size - 1))
            
            #* Is it the fusion model?
            fusion = False
            if metadata != None: fusion = True
            
            #* time
            batch_init_time = time.time()
            
            #* prediction
            data, target = data.to(device), target.to(device)
            #* fusion
            optimizer.zero_grad()
            if fusion:
                metadata = metadata.to(device)
                output = model(data, metadata)
            else: output = model(data)
            
            loss = criterion(output, target)
            
            running_loss.append(float(loss))
            
            #* accuracy
            #* view_as ->, target.view_as(pred) reshapes the target tensor to have the same shape as the pred tensor.
            pred = output.argmax(1, keepdim=True)
            target = target.view_as(pred)
            num_corrects = pred.eq(target).sum().item()
            accuracy = num_corrects / data.size(0) #* corrects/total
            
            running_accuracy.append(float(accuracy))
            
            #* get the 10 worst predictions for this batch
            num_classes = output.shape[-1]
            target_onehot = F.one_hot(target, num_classes=num_classes).float()
            target_onehot = target_onehot.view_as(output)
            
            losses = F.cross_entropy(input=output, target=target_onehot, reduction='none')
            
            k = 10 if len(data) >= 10 else len(data)
            worst_losses, worst_indices = torch.topk(losses, k=k, largest=True)
            
            #* time
            batch_final_time = time.time()
            batch_elapsed_time = batch_final_time - batch_init_time
            running_time += batch_elapsed_time
            total_running_time = batch_final_time - start_time
            
            #* running metrics
            pred = pred.view(-1).numpy(force=True).tolist()
            target = target.view(-1).numpy(force=True).tolist()
            y_pred_accumulator += pred
            y_target_accumulator += target
            
            r_acc, r_pre, r_rec, r_f1, _, _ = calculate_metrics(y_true=y_target_accumulator, y_pred=y_pred_accumulator, classes=classes, labels=labels)
            #* current metrics
            acc, pre, rec, f1, _, _ = calculate_metrics(y_true=target, y_pred=pred, classes=classes, labels=labels)
            
            bar.update(1)
            bar.set_postfix({
                "loss":"{:.3f}".format(float(loss)),
                "acc1":"{:.3f}".format(float(accuracy)),
                "acc2":"{:.3f}".format(acc),
                "pre":"{:.3f}".format(pre),
                "rec":"{:.3f}".format(rec),
                "f1":"{:.3f}".format(f1),
                "r_acc":"{:.3f}".format(r_acc),
                "r_pre":"{:.3f}".format(r_pre),
                "r_rec":"{:.3f}".format(r_rec),
                "r_f1":"{:.3f}".format(r_f1),
                "b_time":convert_seconds(batch_elapsed_time),
                # "running_time":convert_seconds(running_time),
                "t_time":convert_seconds(total_running_time)
                })
            
            for w_idx in worst_indices:
                for k, v in worst_elements.items():
                    if v["loss"] < losses[w_idx]:
                        v["loss"] = float(losses[w_idx])
                        v["path"] = info[0][w_idx]
                        v["id"] = info[1][w_idx]
                        worst_elements = dict(sorted(worst_elements.items(), key=lambda x: x[1]["loss"]))
                        break
            
            loss.backward()
            optimizer.step()
    
    f_acc, f_pre, f_rec, f_f1, _, _ = calculate_metrics(y_true=y_target_accumulator, y_pred=y_pred_accumulator, classes=classes, labels=labels)
    #* acho que é necessário aqui 
    worst_elements = dict(sorted(worst_elements.items(), key=lambda x: x[1]["loss"], reverse=True))
    
    return_dict = {
        "train_loss": mean(running_loss),
        "train_acc1": mean(running_accuracy),
        "train_acc2": f_acc,
        "train_precision": f_pre,
        "train_recall": f_rec,
        "train_f1score": f_f1,
        "reverse_sort_worst_elements": worst_elements
    }
    
    return return_dict

def test_model(model, loader, criterion, epoch_info, start_time, device=torch.device('cuda')):
    model.eval()
    
    y_pred_accumulator = []
    y_target_accumulator =[]
    
    running_loss = []
    running_accuracy = []
    running_time = 0.0
    
    #* worst losses
    worst_elements = {"key_"+str(x):{"loss": 0.0, "path": "", "id": 0} for x in range(10)}
    
    #* class and names
    name_to_class_map = loader.dataset.name_to_class_map
    classes = [k for k, v in name_to_class_map.items()]
    labels = [v for k, v in name_to_class_map.items()]
    
    #* epoch
    # current_epoch = epoch_info[0] if isinstance(epoch_info[0], str) else epoch_info[0] + 1
    
    loader_size = len(loader)
    bar_range = range(loader_size)
    with tqdm.tqdm(bar_range, unit="batchs", position=1, dynamic_ncols=True) as bar:
        with torch.no_grad():
            for batch_idx, (data, info, metadata, target) in enumerate(loader):
                
                #* bar
                bar.set_description("Test Epoch-{}/{} Batch-{}/{}".format(epoch_info[0], epoch_info[1] - 1, batch_idx, loader_size - 1))
                
                #* Is it the fusion model?
                fusion = False
                if metadata != None: fusion = True
                
                #* time
                batch_init_time = time.time()
                
                #* prediction
                data, target = data.to(device), target.to(device)
                #* fusion
                if fusion: 
                    metadata = metadata.to(device)
                    output = model(data, metadata)
                else: output = model(data)
                
                #* loss calculation
                loss = criterion(output, target)
                running_loss.append(float(loss))
                
                # # view_as ->, target.view_as(pred) reshapes the target tensor to have the same shape as the pred tensor.
                # calculo na mão de acurácio - não o mesmo calculo feito para loss. Portanto é normal que os resultados sejam diferentes.
                pred = output.argmax(1, keepdim=True)
                num_corrects = pred.eq(target.view_as(pred)).sum().item()
                accuracy = num_corrects / data.size(0)
                running_accuracy.append(float(accuracy))
                
                #* get the 10 worst predictions for this batch
                num_classes = output.shape[-1]
                target_onehot = F.one_hot(target, num_classes=num_classes).float()
                target_onehot = target_onehot.view_as(output)
                
                losses = F.cross_entropy(input=output, target=target_onehot, reduction='none')
                
                k = 10 if len(data) >= 10 else len(data)
                worst_losses, worst_indices = torch.topk(losses, k=k, largest=True)
                
                #* time
                batch_final_time = time.time()
                batch_elapsed_time = batch_final_time - batch_init_time
                running_time += batch_elapsed_time
                total_running_time = batch_final_time - start_time
                
                #* running metrics
                pred = pred.view(-1).numpy(force=True).tolist()
                target = target.view(-1).numpy(force=True).tolist()
                y_pred_accumulator += pred
                y_target_accumulator += target
                
                r_acc, r_pre, r_rec, r_f1, _, _ = calculate_metrics(y_true=y_target_accumulator, y_pred=y_pred_accumulator, classes=classes, labels=labels)
                #* current metrics
                acc, pre, rec, f1, _, _ = calculate_metrics(y_true=target, y_pred=pred, classes=classes, labels=labels)
                
                #* bar
                bar.update(1)
                bar.set_postfix({
                    "loss":"{:.3f}".format(float(loss)),
                    "acc1":"{:.3f}".format(float(accuracy)),
                    "acc2":"{:.3f}".format(acc),
                    "pre":"{:.3f}".format(pre),
                    "rec":"{:.3f}".format(rec),
                    "f1":"{:.3f}".format(f1),
                    "r_acc":"{:.3f}".format(r_acc),
                    "r_pre":"{:.3f}".format(r_pre),
                    "r_rec":"{:.3f}".format(r_rec),
                    "r_f1":"{:.3f}".format(r_f1),
                    "b_time":convert_seconds(batch_elapsed_time),
                    # "running_time":convert_seconds(running_time),
                    "t_time":convert_seconds(total_running_time)
                    })
                
                for w_idx in worst_indices:
                    for k, v in worst_elements.items():
                        if v["loss"] < losses[w_idx]:
                            v["loss"] = float(losses[w_idx])
                            v["path"] = info[0][w_idx]
                            v["id"] = info[1][w_idx]
                            worst_elements = dict(sorted(worst_elements.items(), key=lambda x: x[1]["loss"]))
                            break
    
    f_acc, f_pre, f_rec, f_f1, f_classification_report, f_confusion_matrix = calculate_metrics(y_true=y_target_accumulator, y_pred=y_pred_accumulator, classes=classes, labels=labels)
    worst_elements = dict(sorted(worst_elements.items(), key=lambda x: x[1]["loss"], reverse=True))
    
    return_dict = {
        "test_loss": mean(running_loss),
        "test_acc1": mean(running_accuracy),
        "test_acc2": f_acc,
        "test_precision": f_pre,
        "test_recall": f_rec,
        "test_f1score": f_f1,
        "test_classification_report_dict": f_classification_report[0],
        "test_classification_report_str": f_classification_report[1],
        "test_confusion_matrix": f_confusion_matrix.tolist(),
        "test_targets": y_target_accumulator,
        "test_predictions": y_pred_accumulator,
        "reverse_sort_worst_elements": worst_elements
    }
    
    return return_dict