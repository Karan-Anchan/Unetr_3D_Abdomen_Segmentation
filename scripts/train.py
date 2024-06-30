import torch
from src.data_loading import get_data_loaders
from src.model import create_model
from src.training import train

def main():
    
    """
    Main function to train the UNETR model.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./data"
    train_loader = get_data_loaders(data_dir, batch_size=2, num_workers=4)
    model = create_model().to(device)
    train(model, train_loader, epochs=10, device=device)

if __name__ == "__main__":
    main()