import random, os, torch
import numpy as np
from torch import nn


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeding everything to seed {seed}")
    return


# custom early stopping, based on chosen metric, works for minimizing metrics
class EarlyStopping:
    def __init__(self, patience=5, delta=1e-6):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif current_score > self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = current_score
            self.counter = 0


class NMFModel(nn.Module):
  def __init__(self, input_dim, num_classes):
    super(NMFModel, self).__init__()
    self.fc1 = nn.Linear(input_dim, 128)
    self.relu1 = nn.ReLU()
    self.dropout1 = nn.Dropout(p=0.5)
    self.fc2 = nn.Linear(128, num_classes)

  def forward(self, x):
    x = self.fc1(x)
    x = self.relu1(x)
    x = self.dropout1(x)
    x = self.fc2(x)
    return x


def get_model(in_features, num_classes):
    model = NMFModel(input_dim=in_features, num_classes=num_classes)
    return model


def save_model(model, encoder, epoch, num_classes, in_features, lr, lr_factor, save_path, fname):
    save_dict = {"model_state_dict": model.state_dict(),
                 "encoder": encoder,
                 "num_classes": num_classes,
                 "in_features": in_features,
                 "epoch": epoch,
                 "lr": lr,
                 "lr_scheduler_factor": lr_factor}
    torch.save(save_dict, save_path/f"{fname}.pth")


def load_model(save_path, fname):
    loaded_model = torch.load(
        save_path/f"{fname}.pth", map_location=torch.device("cpu"))
    num_classes = loaded_model["num_classes"]
    in_features = loaded_model["in_features"]
    model = get_model(in_features, num_classes)
    model.load_state_dict(loaded_model["model_state_dict"])
    return model