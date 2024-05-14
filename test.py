from dataset import get_testloader
from utils import seed_all, load_model
from train import validate_batch
from torch.nn import CrossEntropyLoss
from pathlib import Path


if __name__ == "__main__":
    seed_all(42)
    test_data = Path("data/adata_test_uncompressed.h5ad")
    model_save_path = Path("models/")
    model_fname = "4_Niche_classifier"
    bs = 512
    device = "mps"

    test_loader = get_testloader(test_data, batch_size=bs)
    model = load_model(model_save_path, model_fname)
    model = model.to(device)

    f1_s, _ = validate_batch(test_loader, model, CrossEntropyLoss(), device)
    print(f1_s)
