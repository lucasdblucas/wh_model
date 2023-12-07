import torch.nn as nn
import torch

class FusionModel(nn.Module):
    
    def __init__(self, first_column_model: nn.Module, model_fusion_config: dict, *args, **kwargs) -> None:
        super(FusionModel, self).__init__(*args, **kwargs)
        
        #* pre-trained model for images
        #TODO: justo for resnet, maybe change it if there is another architecture that should be used 
        self.first_column_model = first_column_model
        self.fc_input = model_fusion_config["model_architecture"]["num_fc_in_features"]
        self.out_features_first_column = self.fc_input
        
        self.num_classes = model_fusion_config["model_architecture"]["num_classes"]
        
        #* model for metadata
        growth = 12
        self.out_features_second_column = 64*growth
        num_in_features_second_column = len(model_fusion_config["config_dict"]["leprosy_fusion_config"]["metadata_features"]["features_list"])
        default_pDropout = model_fusion_config["config_dict"]["deep_config"]["dropout"]["use"][0]
        
        
        self.metadata_features = nn.Sequential(
            nn.Linear(num_in_features_second_column, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=default_pDropout),
            nn.Linear(64, out_features=self.out_features_second_column),
            nn.ReLU(inplace=True),
            nn.Dropout(p=default_pDropout)
        )
        
        combined_entry_size = (self.out_features_first_column + self.out_features_second_column)
        self.combined_features = nn.Sequential(
            nn.Linear(combined_entry_size, combined_entry_size * growth),
            nn.ReLU(inplace=True),
            nn.Dropout(p=default_pDropout),
            nn.Linear(combined_entry_size * growth, combined_entry_size),
            nn.ReLU(inplace=True),
            # nn.Linear(combined_entry_size, 64),
            nn.Linear(combined_entry_size, out_features=self.num_classes)
        )

        # self.dropout = torch.nn.Dropout(default_pDropout)
        
        
    def forward(self, x, y):
        
        image_features = self.first_column_model(x)
        metadata_features = self.metadata_features(y)
        # print("Num elements on x: {}, y: {}, fc_input: {}".format(image_features.numel(), metadata_features.numel(), self.fc_input))
        
        image_features = image_features.view(-1, self.out_features_first_column)
        metadata_features = metadata_features.view(-1, self.out_features_second_column)
        
        combined = torch.cat(
            [image_features, metadata_features], 
            dim=1
        )
        output = self.combined_features(combined)
        
        return output
        