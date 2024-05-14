import torch, argparse
import numpy as np
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from utils import EarlyStopping, seed_all, get_model, save_model
from dataset import get_data, get_dataloaders


def train_epoch(train_loader, model, optimizer, criterion, epoch, device):
    running_loss = 0.0
    train_l = tqdm(train_loader, total=len(train_loader))
    for X, y in train_l:
        X, y = X.to(device), y.to(device)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_l.set_postfix_str(f'[{epoch + 1}] loss: {running_loss:.3f}')
    return running_loss


def validate_batch(loader, model, criterion, device):
    model = model.to(device)
    model.eval()
    gts = []
    preds = []
    running_loss = 0.0
    with torch.no_grad():
        with tqdm(loader, unit="batch", leave=True) as tepoch:
            for inputs, targets in tepoch:
                inputs = inputs.float().to(device)
                targets = targets.long().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                preds.extend(predictions.cpu().numpy())
                gts.extend(targets.cpu().numpy())
    f1_s = f1_score(np.array(gts), np.array(preds), average="weighted")
    return f1_s, running_loss


def create_train_parser():
    my_parser = argparse.ArgumentParser(description='Script used for training a model')

    my_parser.add_argument('--run_number',
                           type=str,
                           help='Number of this run')

    my_parser.add_argument('--encoder',
                           type=str,
                           help='Name of an encoder',
                           default="custom_classifier")

    my_parser.add_argument('--batch_size',
                           type=int,
                           help='Number of patches in a batch', default=512)

    my_parser.add_argument('--lr',
                           type=float,
                           help='Learning rate', default=1e-3)
    
    my_parser.add_argument('--lr_scheduler_factor',
                           type=float,
                           help='Factor to reduce the learning rate', default=0.5)
    
    my_parser.add_argument('--lr_scheduler_patience',
                           type=int,
                           help='Number of epochs for the learning rate scheduler to wait', default=5)
    
    my_parser.add_argument('--max_epochs',
                           type=int,
                           help='Maximal number of epochs to train for', default=50)

    my_parser.add_argument('--validate_every_n_epochs',
                           type=int,
                           help='Validate every n epochs', default=1)

    my_parser.add_argument('--es_patience',
                           type=int,
                           help='patience for early stopping', default=5)
    
    my_parser.add_argument('--kfold',
                           type=int,
                           help='number of folds for k-fold cross-validation', default=5)

    args = my_parser.parse_args()
    return args


if __name__ == "__main__":
    seed_all(seed=42)
    device = torch.device('mps')
    args = create_train_parser()

    train_data = Path("data/adata_trainval_uncompressed.h5ad")
    model_save_path = Path("./models")
    model_save_path.mkdir(parents=True, exist_ok=True)
    log_path = Path("./logs")
    log_path.mkdir(parents=True, exist_ok=True)
    fname = f"{args.run_number}_{args.encoder}"
    val_log = []

    total_loss = np.zeros(args.kfold)
    kfold = StratifiedKFold(n_splits=args.kfold) # automatically shuffled during dataloader creating and numpy seed is already set to 42 
    data, labels, labels_map = get_data(train_data)
    num_classes, in_features = len(labels_map), data[0].shape[0]
    best_model = [np.inf, None]

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data, labels)):
        print(f"kfold: {fold+1}/{args.kfold}")
        model = get_model(in_features=in_features, num_classes=num_classes)
        early_stop = EarlyStopping(patience=args.es_patience)
        criterion = CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", verbose=True, factor=args.lr_scheduler_factor, min_lr=1e-5, patience=args.lr_scheduler_patience)
        model = model.to(device)
        train_loader, val_loader = get_dataloaders(data, labels, train_idx, val_idx, batch_size=args.batch_size)

        for epoch in range(args.max_epochs):  # loop over the dataset
            train_loss = train_epoch(train_loader, model, optimizer, criterion, epoch, device)

            if epoch % args.validate_every_n_epochs == args.validate_every_n_epochs-1:
                f1_s, val_loss = validate_batch(val_loader, model, criterion, device)
                val_log.append([fold, epoch, train_loss, val_loss, f1_s])
                early_stop(val_loss)
                scheduler.step(metrics=val_loss)

                if early_stop.should_stop:
                    print(f"Stopped early at epoch: {epoch+1}")
                    total_loss[fold] = early_stop.best_score

                    # module to find best model for saving
                    # technically retraining on entire trainval required but this is just an exercise
                    if early_stop.best_score < best_model[0]:
                        best_model[0] = early_stop.best_score
                        best_model[1] = model
                    break
            
    # epoch parameter does not represent the real epoch of corresponding kfold but the last fold's epoch. Since variable to utilized it does not matter anyway
    save_model(best_model[1], args.encoder, epoch, num_classes, in_features, args.lr, args.lr_scheduler_factor, save_path=model_save_path, fname=fname)
    val_log.append([0, 0, 0, np.mean(total_loss), 0])
    val_log = np.asarray(val_log)
    np.savetxt(f"{str(log_path)}/{fname}.csv", val_log, delimiter=",", fmt=["%d", "%d", "%.5f", "%.5f", "%.5f"])
    
    print(f"total_loss across all folds: {total_loss}")
    print(f"total validation loss: {np.mean(total_loss)}")