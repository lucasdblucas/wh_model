import torch
from torchvision.models import resnet50

class ResNet50MultiFC(torch.nn.Module):
    def __init__(self, num_outputs):
        super(ResNet50MultiFC).__init__()
        
        model = resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        
        fc_list = []
        for _ in range(num_outputs):
            fc_list.append(torch.nn.Linear(num_ftrs, 1))
        
        model.fc = None
        self.model = model
        self.last_fcs = torch.nn.ModuleList(fc_list)

    def forward(self, x):
        x = self.model(x)
        
        outputs = []
        for fc in self.lst_fcs:
            outputs.append(fc(x))
        
        return (*outputs,)
    
def wh_split_resnet_50(**kwargs):
    transform_config = kwargs["current_project_info"]["data_config"]["transform_config"]
    deep_config = kwargs["current_project_info"]["train_infe_config"]["deep_config"]
    task_config = kwargs["current_project_info"]["task_config"]
    regression_config = task_config["regression_config"]
    
    num_classes = len(regression_config["targets"]["use"])
    grayscale = transform_config["grayscale"]["use"]
    img_channel = 1 if grayscale else 3
    dropout = deep_config["dropout"]["use"][0] #* the first from the list
    
    # return WideResNetTwoOutputs(
    #     img_channel=img_channel, 
    #     depth=16,
    #     num_classes=num_classes,
    #     widen_factor=8, 
    #     dropRate=dropout
    # )

def wh_renet_50(**kwargs):
    transform_config = kwargs["current_project_info"]["data_config"]["transform_config"]
    # deep_config = kwargs["current_project_info"]["train_infe_config"]["deep_config"]
    task_config = kwargs["current_project_info"]["task_config"]
    regression_config = task_config["regression_config"]
    
    grayscale = transform_config["grayscale"]["use"]
    num_channel = 1 if grayscale else 3
    # dropout = deep_config["dropout"]["use"][0] #* the first from the list
    num_outputs = len(regression_config["targets"]["use"])
    
    return ResNet50MultiFC(
        num_channel=num_channel, 
        num_outputs=num_outputs
    )