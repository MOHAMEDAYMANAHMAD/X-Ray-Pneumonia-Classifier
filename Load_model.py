from torchvision import models
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"



def get_model(model_name, path):

    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = torch.nn.Linear(512, 2)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = torch.nn.Linear(1280, 2)
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = torch.nn.Linear(model.classifier.in_features, 2)
    else:
        raise ValueError("Unknown model")
    
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)   
    model.eval()
    return model



all_models = {
    "resnet18": get_model("resnet18","models/resnet18.pth"),
    "densenet121": get_model("densenet121","models/densenet121.pth"),
    "efficientnet_b0": get_model("efficientnet_b0","models/efficientnet_b0.pth")
}

