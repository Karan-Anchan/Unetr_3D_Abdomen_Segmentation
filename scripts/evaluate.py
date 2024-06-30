import torch
from src.data_loading import get_data_loaders
from src.model import create_model
from src.evaluation import evaluate

def main():
    
    """
    Main function to evaluate the UNETR model.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "./data"
    eval_loader = get_data_loaders(data_dir, batch_size=1, num_workers=2)
    model = create_model().to(device)
    model.load_state_dict(torch.load('model.pth'))  # Load pre-trained model
    evaluate(model, eval_loader, device=device)

if __name__ == "__main__":
    main()
