import torch
from torchvision import  transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_image(model, img):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)
    classes = ["NORMAL", "PNEUMONIA"]
    return classes[pred.item()], conf.item()



