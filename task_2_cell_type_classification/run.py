import scanpy as sc
import utils
import model
import train

# the count matrix is kept as is, no further preprocessing applied
adata = sc.read_h5ad("C://backup//Bachelorarbeit//adata_processed_condition.h5ad")

# prepare data sets
X = adata.to_df()
Y = adata.obs["cell_type"].cat.codes.to_frame()
dataset = utils.df_to_tensor_dataset(X, Y)
dataloaders = utils.create_data_loaders(dataset)

# plot label distribution
utils.plot_label_distribution(Y)

# get random baseline estimates
NUM_CLASSES = len(adata.obs['cell_type'].unique())
random_baseline = utils.random_baseline(dataloaders['test'], NUM_CLASSES)
categorical_baseline = utils.categorical_baseline(dataloaders['test'], dataloaders["train"], NUM_CLASSES)

# train and test logistic regression model
logreg = model.LogisticRegression(X.shape[1], NUM_CLASSES)
logreg_loss = train.fit(logreg, dataloaders, 0.01, 0.9, 100)
logreg_results = utils.evaluate_model(logreg, dataloaders['test'], 'cuda')
utils.plot_losses(logreg_loss)

# train and test logistic regression model
nn = model.NeuralNetwork(X.shape[1], 256, NUM_CLASSES)
nn_loss = train.fit(nn, dataloaders, 0.01, 0.9, 100)
nn_results = utils.evaluate_model(nn, dataloaders['test'], 'cuda')
utils.plot_losses(nn_loss)

# plot the performance metrics of baselines and models
utils.plot_metrics({"random baseline":random_baseline, "categorical baseline":categorical_baseline, "logistic regression":logreg_results, "neural network":nn_results})

# note: for gridsearch one would need to iterate over hyperparameters and compare losses, then pick best final model for test set evaluation
# for example sake this is omitted here
# the plots correspond to the models trained in this script