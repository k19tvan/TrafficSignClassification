from function import ImagePrediction
from basic_model import BasicModel
import torch

if __name__ == "__main__":
    
    try:
        model = BasicModel()
        model.load_state_dict(torch.load("model_weights.pth"))
        model.eval()
        
        image_path = "/home/enn/workspace/deep_learning/Traffic_Sign_Classification/CS231-TrafficSignClassification/Data/instruction/005.png"
        ImagePrediction(model, image_path)
        
    except:
        print("There wasn't model trained")
