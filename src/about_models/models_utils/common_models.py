from torchvision.models import resnet50, resnet101, resnet34
from torch.hub import load
import pretrainedmodels as ptm

class CommonModels():
    models_type: dict = {
        "resnet34": lambda: resnet34(pretrained=False),
        "resnet50": lambda: resnet50(pretrained=False),
        "resnet101": lambda: resnet101(pretrained=False),
        "vgg13": lambda: load(repo_or_dir='pytorch/vision:v0.10.0', model="vgg13", pretrained=False),
        "vgg13_bn": lambda: load(repo_or_dir='pytorch/vision:v0.10.0', model="vgg13_bn", pretrained=False),
        "vgg16": lambda: load(repo_or_dir='pytorch/vision:v0.10.0', model="vgg16", pretrained=False),
        "vgg16_bn": lambda: load(repo_or_dir='pytorch/vision:v0.10.0', model="vgg16_bn", pretrained=False),
        "vgg19": lambda: load(repo_or_dir='pytorch/vision:v0.10.0', model="vgg19", pretrained=False),
        "vgg19_bn": lambda: load(repo_or_dir='pytorch/vision:v0.10.0', model="vgg19_bn", pretrained=False),
        "inceptionv3": lambda: load(repo_or_dir='pytorch/vision:v0.10.0', model="inception_v3", pretrained=False),
        "inceptionv4": lambda num_classes: ptm.inceptionv4(num_classes=num_classes, pretrained=False)
    }