import numpy as np
import pickle

from matplotlib import pyplot as plt


num_epochs = 10

# Load data from saved file
list_hist = pickle.load(open("list_hist.pkl", "rb"))

# Plot the training curves of validation accuracy vs. number of training epochs for each fold training
list_fold_hist = [[h.cpu().numpy() for h in hist] for hist in list_hist]
plt.title("Validation Accuracy vs. Number of Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
for i, fhist in enumerate(list_fold_hist):
    plt.plot(range(1, num_epochs + 1), fhist, label="Fold " + str(i))
plt.ylim((0, 1.))
plt.xticks(np.arange(1, num_epochs + 1, 1.0))
plt.legend()
plt.show()
