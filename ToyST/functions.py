import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scanpy as sc
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import matthews_corrcoef, accuracy_score, make_scorer, log_loss, confusion_matrix
from skorch import NeuralNetClassifier
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from  collections import Counter
import pickle

class Network(nn.Module): # model class
    def __init__(self, in_layers, hid_layers1, hid_layers2, hid_layers3, out_layers):
        super().__init__()
        self.fc1 = nn.Linear(in_layers, hid_layers1)
        self.fc2 = nn.Linear(hid_layers1, hid_layers2)
        self.fc3 = nn.Linear(hid_layers2, hid_layers3)
        self.out = nn.Linear(hid_layers3, out_layers)

    def forward(self, x):
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.relu(self.fc3(x))
        x=F.softmax(self.out(x), dim=1)

        return x

def save_to_file(model, file_name, grid=False): # function for saving model
    if grid:
        with open(file_name, 'wb') as f:
            pickle.dump(model, f)
    else:
        torch.save(model.state_dict(), file_name)

class h5ad_Dataset: # dataset class
    def __init__(self, path_to_data, label_col):
        self.path_to_data, self.label_col = path_to_data, label_col
        self.labels, self.classes, self.labels_to_classes, self.classes_to_labels, self.counts = self.make_dataset(self.path_to_data, self.label_col) # attributes of the model

    # labels -- string labels
    # classes -- labels presented as integer
    # labels_to_classes -- dictionary to change from labels to classes
    # classes_to_labels -- dictionary to change from classes to labels

    @staticmethod
    def make_dataset(path_to_data, label_col): # creating dataset
        print('reading in data')
        adata = sc.read_h5ad(path_to_data)
        adata = adata[:, adata.var.index[adata.var['highly_variable']]] # filter data for only highly variable genes -- paper only used higly variable genes for downstream + able to run on my laptop :)
        print('filtered')
        print(adata.shape) # checking if filtering worked

        counts = torch.from_numpy(adata.X.toarray()) # producing torch from these counts
        labels = adata.obs[label_col]

        unique_labels = list(set(labels))
        # create dictionaries to change inbetween
        labels_to_classes = {unique_labels[i]: i for i in range(len(unique_labels))}
        classes_to_labels = {value: key for key, value in labels_to_classes.items()}

        classes = torch.as_tensor([labels_to_classes[label] for label in labels]) # producing torch from classes
        print('done')

        assert len(classes) == len(counts)
        return labels, classes, labels_to_classes, classes_to_labels, counts

    def __len__(self):
        return(len(self.classes))

    def __getitem__(self, index):
        return self.labels[index], self.counts[index]


class EarlyStopping: # Early stopping to break out of epoch loop when loss doesn't improve over x number of epochs (patience)
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0

    def should_stop(self):
        return self.early_stop

def crossValidation(n_splits, shuffle, random_state, model, dataset, batch_size, epochs, learn_r=0.001): # training with cross validation
    sf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    best_model = None
    best_val_loss = float('inf')

    losses_train, mccs_train, accs_train = [], [], []
    losses_val, mccs_val, accs_val = [], [], []

    print('starting cross validation')
    for fold_i, (train_ids, val_ids) in enumerate(sf.split(X=dataset[0], y=dataset[1])): # iterate through the crossvalidation split
        print(f'Fold: {fold_i+1}')

        X_train, X_val, y_train, y_val = dataset[0][train_ids], dataset[0][val_ids], dataset[1][train_ids], dataset[1][val_ids] # get training and validation set from split

        train_dataset = TensorDataset(torch.as_tensor(X_train), torch.as_tensor(y_train))
        val_dataset = TensorDataset(torch.as_tensor(X_val), torch.as_tensor(y_val))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=learn_r)
        criterion = nn.CrossEntropyLoss()

        early_stopping = EarlyStopping(patience=5, verbose=True)
        for epoch in range(epochs):
            model.train() # training model
            running_loss = 0.0
            mcc = 0.0
            accs = 0.0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                _, predicted = torch.max(outputs, dim=1)
                loss = criterion(outputs, targets)

               # print(predicted)
               # print(targets)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                mcc += matthews_corrcoef(targets, predicted)
                accs += accuracy_score(targets, predicted)

            # remember scores per epoch
            losses_train.append(running_loss / len(train_loader))
            mccs_train.append(mcc / len(train_loader))
            accs_train.append(accs / len(train_loader))

            print(f"Fold {fold_i+1}/{n_splits}, Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                running_loss = 0.0
                mcc = 0.0
                accs = 0.0
                for inputs, targets in val_loader:
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs, dim=1)
                    val_loss += criterion(outputs, targets).item()

                    running_loss += loss.item()
                    mcc += matthews_corrcoef(targets, predicted)
                    accs += accuracy_score(targets, predicted)

            # remember validation scores
            losses_val.append(running_loss / len(val_loader))
            mccs_val.append(mcc / len(val_loader))
            accs_val.append(accs / len(val_loader))

            avg_val_loss = val_loss / len(val_loader)
            print(f"Fold {fold_i + 1}/{n_splits}, Validation Loss: {avg_val_loss}")

            early_stopping(avg_val_loss)
            if early_stopping.should_stop():
                print("Early stopping triggered.")
                break

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model

    print(f"Best validation loss: {best_val_loss}")
    return best_model, [losses_train, mccs_train, accs_train], [losses_val, mccs_val, accs_val]

def crossValidationGrid(n_splits, shuffle, random_state, module, dataset, batch_size, epochs, labels): # cross validation training for optimisation of hyper-parameters
    sf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    model = NeuralNetClassifier(module=module,max_epochs=epochs,batch_size=batch_size)
    scoring = { # use MCC to check in case of imbalanced dataset, also to compare to test set which is imbalanced
        "Loss": make_scorer(log_loss, needs_proba=True, labels=labels),
        "MCC": make_scorer(matthews_corrcoef),
        "Acc": make_scorer(accuracy_score)}

    param_grid = { # optimised parameters (for now only learning rate)
        'optimizer__lr': [0.001, 0.01, 0.1]
    }

    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=sf, scoring =scoring, return_train_score=True, refit='MCC', verbose=0)
    grid.fit(dataset[0], dataset[1])

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    scores_grid = grid.cv_results_
    i = grid.best_index_

    losses_train, mccs_train, accs_train = [],[],[]
    losses_val, mccs_val, accs_val = [],[],[]

    for fold in range(n_splits): # get scores for the best performing model
        losses_train.append(scores_grid[f'split{fold}_train_Loss'][i])
        mccs_train.append(scores_grid[f'split{fold}_train_MCC'][i])
        accs_train.append(scores_grid[f'split{fold}_train_Acc'][i])

        losses_val.append(scores_grid[f'split{fold}_test_Loss'][i])
        mccs_val.append(scores_grid[f'split{fold}_test_MCC'][i])
        accs_val.append(scores_grid[f'split{fold}_test_Acc'][i])

    return best_model, [losses_train, mccs_train, accs_train], [losses_val, mccs_val, accs_val]

def bootstrap(true, predicted, n): # function for getting bootstrapped scores to produce confidence intervals
    accs = []
    mccs = []
    for i in range(n):
        # Randomly sample with replacement
        indices = np.random.choice(len(true), size=len(true), replace=True)

        # Create bootstrap samples
        true_bootstrap = [true[i] for i in indices]
        predicted_bootstrap = [predicted[i] for i in indices]

        # Calculate metric for bootstrap sample
        accs.append(accuracy_score(true_bootstrap, predicted_bootstrap)*100)
        mccs.append(matthews_corrcoef(true_bootstrap, predicted_bootstrap)*100)
        #f1s.append(f1_score(true_labels_bootstrap, predicted_labels_bootstrap, average='micro')*100)
    # Analyze the distribution of the metric (e.g., calculate confidence intervals)
    return mccs, accs

def testing(test_X, model, grid = False):
    test_X = torch.as_tensor(test_X, dtype=torch.float32)
    if(grid): # grid model from skorch uses different function to make predictions
        outputs = torch.as_tensor(model.predict_proba(test_X))
        _, predicted = torch.max(outputs, dim=1)
    else:
        model.eval()  # set the model to evaluation mode
        outputs = model(test_X)
        # check if dimensions is greater than 1
        if len(outputs.shape) > 1:
            _, predicted = torch.max(outputs, dim=1)
        else:
            predicted = outputs
    #print(predicted)
    return predicted

def plot_metrics(train_loss, val_loss, train_metrics, val_metrics, metric_names, x_axis, colors = ['orange', '#1f77b4'], file=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    ax1.minorticks_on()
    ax2.minorticks_on()


    #lines = [i * (len(loss) // n_folds) for i in range(n_folds)]

    for i, metric in enumerate(train_metrics):
        ax1.plot(range(len(train_metrics[0])), metric, linestyle = 'dashed', c = colors[i], label = f'Train {metric_names[i]}')
        ax1.set_ylabel("Metric")
        ax1.set_xlabel(x_axis)
    for i, metric in enumerate(val_metrics):
        ax1.plot(range(len(val_metrics[0])), metric, c=colors[i], label = f'Val {metric_names[i]}')
    ax1.grid(which='major')
    ax1.grid(which='minor', linestyle=":")
    ax1.set_title('Metrics learning curves')
    ax1.set_ylim(0,1)

    ax2.plot(range(len(train_loss)), train_loss, linestyle = 'dashed', c='darkred')
    ax2.plot(range(len(val_loss)), val_loss, c='darkred')

    ax2.set_ylabel("Loss")
    ax2.set_xlabel(x_axis)
    ax2.grid(which='major')
    ax2.grid(which='minor', linestyle=":")
    ax2.set_title('Loss learning curve')
    #for line in lines:
    #    ax1.axvline(line, color='red', linestyle='--')
    #    ax2.axvline(line, color='red', linestyle='--')
    if file != None:
        plt.savefig(file)
    plt.show()

def get_confidence(list): # calculating confidence interval
    se = np.std(list)
    x = scipy.stats.t.ppf((1 + 0.95 ) / 2, len(list))
    return se*x

def create_plot_allperformance(bar_source, labels, colors, file=None): # comparing performace of the two models

    barWidth = 0.3
    bars, cons, pos = [], [], []

    for i, score in enumerate(bar_source):
        bars.append([np.mean(score[0]), np.mean(score[1])])
        cons.append([get_confidence(score[0]), get_confidence(score[1])])
        positions = [x + barWidth*i for x in np.arange(len(bars[0]))]
        pos.append(positions)

    plt.minorticks_on()
    # Create bars
    for i in range(len(bars)):
        plt.bar(pos[i], bars[i], width = barWidth, color = colors[i], alpha = 0.5, yerr=cons[i], capsize=7, label=labels[i])

    # general layout
    plt.xticks([r + 0.5*barWidth for r in range(len(bars[0]))], [ 'MCC', 'Acc'])
    plt.ylabel('score')
    plt.legend()
    if file != None:
        plt.savefig(file)
    plt.show()

def plot_heatmap(true, pred, title='', file=None):
    counted = Counter(true)
    labels = [value for value, count in counted.most_common()]
    s_labels = [label[:8]+'...' for label in labels]
    cm = confusion_matrix(true, pred, labels=labels)
    # calculate row sums (for calculating % & plot annotations)
    cm_sum = cm.sum(axis=1)[:, np.newaxis]
    # calculate proportions
    cm_perc = cm / cm_sum.astype(float) * 100
    cm_perc = np.nan_to_num(cm_perc, copy=True)
    # empty array for holding annotations for each cell in the heatmap
    annot = np.empty_like(cm).astype(str)
    # get the dimensions
    nrows, ncols = cm.shape
    # cycle over cells and create annotations for each cell
    for i in range(nrows):
        for j in range(ncols):
            # get the count for the cell
            c = cm[i, j]
            # get the percentage for the cell
            p = cm_perc[i, j]
            s = cm_sum[i]
            if p != 0:
                annot[i, j] = f'{c}/{int(s)}\n{int(p)}%'
            else:
                annot[i, j] = ""
    # convert the array to a dataframe. To plot by proportion instead of number, use cm_perc in the DataFrame instead of cm
    cm_perc = pd.DataFrame(cm_perc, index=labels, columns=labels)
    cm_perc.index.name = 'Actual'
    cm_perc.columns.name = 'Predicted'
    # create empty figure with a specified size
    fig, ax = plt.subplots(figsize=(10.5,11))
    # plot the data using the Pandas dataframe. To change the color map, add cmap=..., e.g. cmap = 'rocket_r'
    sns.set(font_scale=1.3)
    sns.heatmap(cm_perc, annot=annot, fmt='', ax=ax, xticklabels=s_labels,
                yticklabels=s_labels,
                cmap='Blues') #, annot_kws={"size": 14
    plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    #plt.figure(figsize=(11, 12))

    if file != None:
        plt.savefig(file)

    plt.show()
