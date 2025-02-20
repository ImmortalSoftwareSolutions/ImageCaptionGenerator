import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import sys


# Load model
class ViTClassifier(torch.nn.Module):
    def __init__(self, model):
        super(ViTClassifier, self).__init__()
        self.vit = model
    
    def forward(self, img):
        vit_image_classification_output = self.vit(img)
        logits = vit_image_classification_output.logits
        return logits


# Load tag dictionary
with open("tags.json", "r") as fp:
    tag_dict = json.load(fp)
ind_to_tag = {v: k for k, v in tag_dict.items()}


# Load the trained model
model_new = torch.jit.load("hashtag_generator_model.pt", map_location=torch.device("cpu"))
model_new.eval()

def predict_hashtags(image_path, topk=10):
    """Predicts hashtags for a given image file."""
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_new.to(device)
    model_new.eval()
    
    transforms_list = transforms.Compose([
        transforms.Resize(size=(224, 224)),
        transforms.ToTensor()
    ])
    
    img = Image.open(image_path).convert("RGB")
    img = transforms_list(img)
    img = img.unsqueeze(dim=0).to(device)
    
    with torch.no_grad():
        logits = model_new(img)
    
    scores, tags_present = logits.squeeze().topk(topk)
    tags = ["#" + str(ind_to_tag[ind.item()]) for ind in tags_present]
    print(" ".join(tags))
    
    return " ".join(tags)

#img_path = 'C:/Users/bossu/OneDrive/Desktop/instabot/ImageCaptionGenerator/New folder/Images/179009558_69be522c63.jpg'
#hashtags = predict_hashtags(img_path)
#print("---------Hashtags------")
#print(hashtags)