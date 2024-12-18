from basic_model import BasicModel
from function import DataLoaderCreate, ModelTraining

if __name__ == "__main__":
    
    train_loader, val_loader, test_loader = DataLoaderCreate()
    model = BasicModel()

    ModelTraining(model, train_loader, test_loader)
