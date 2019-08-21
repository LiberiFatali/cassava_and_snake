from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from backbone import initialize_model
from helper import train_model


print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "data/train"
# data_dir = ''

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# model_name = "resnet-152"
model_name = "se_resnext101_32x4d"

# Number of classes in the dataset
# num_classes = 5
num_classes = 45

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 500
num_epoch_to_stop_if_no_better = 50

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

#
checkpoint_save_path = './pytorch_space/snake_se_resnext101_32x4d.pth'


# Set Model Parameters’ .requires_grad attribute
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# This helper function sets the ``.requires_grad`` attribute of the
# parameters in the model to False when we are feature extracting. By
# default, when we load a pretrained model all of the parameters have
# ``.requires_grad=True``, which is fine if we are training from scratch
# or finetuning. However, if we are feature extracting and only want to
# compute gradients for the newly initialized layer then we want all of
# the other parameters to not require gradients. This will make more sense
# later.
#
#
#


# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
print(model_ft)


# ### Custom dataset

class CassavaDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


# init_dataset = TensorDataset(
#     torch.randn(100, 3, 24, 24),
#     torch.randint(0, 10, (100,))
# )

# lengths = [int(len(init_dataset)*0.8), int(len(init_dataset)*0.2)]
# subsetA, subsetB = random_split(init_dataset, lengths)
# datasetA = MyDataset(
#     subsetA, transform=transforms.Normalize((0., 0., 0.), (0.5, 0.5, 0.5))
# )
# datasetB = MyDataset(
#     subsetB, transform=transforms.Normalize((0., 0., 0.), (0.5, 0.5, 0.5))
# )


# Load Data
# ---------
#
# Now that we know what the input size must be, we can initialize the data
# transforms, image datasets, and the dataloaders. Notice, the models were
# pretrained with the hard-coded normalization values, as described
# `here <https://pytorch.org/docs/master/torchvision/models.html>`__.
#
#
#


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    # 'train': transforms.Compose([
    #     transforms.RandomResizedCrop(size=input_size, scale=(0.8, 1.0)),
    #     transforms.RandomRotation(degrees=15),
    #     transforms.ColorJitter(),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.CenterCrop(size=input_size),  # Image net standards
    #     transforms.ToTensor(),
    #     transforms.Normalize([0.485, 0.456, 0.406],
    #                          [0.229, 0.224, 0.225])  # Imagenet standards
    # ]),
    'train': transforms.Compose([
        transforms.Resize(input_size + 20),
        transforms.RandomRotation(15, expand=True),
        transforms.RandomCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

print("Initializing Datasets and Dataloaders...")

# Load full set
image_datasets = {x: datasets.ImageFolder(data_dir, data_transforms[x]) for x in ['train', 'val']}

# # Split training and validation datasets
# full_dataset = datasets.ImageFolder(data_dir)
# train_size = int(0.80 * len(full_dataset))
# val_size = len(full_dataset) - train_size
# train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
#
# train_dataset = CassavaDataset(train_dataset, transform=data_transforms['train'])
# val_dataset = CassavaDataset(val_dataset, transform=data_transforms['val'])
# image_datasets = {
#     'train': train_dataset,
#     'val': val_dataset
# }

# Create training and validation dataloaders
dataloaders_dict = {
    x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in
    ['train', 'val']}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create the Optimizer
# --------------------
#
# Now that the model structure is correct, the final step for finetuning
# and feature extracting is to create an optimizer that only updates the
# desired parameters. Recall that after loading the pretrained model, but
# before reshaping, if ``feature_extract=True`` we manually set all of the
# parameter’s ``.requires_grad`` attributes to False. Then the
# reinitialized layer’s parameters have ``.requires_grad=True`` by
# default. So now we know that *all parameters that have
# .requires_grad=True should be optimized.* Next, we make a list of such
# parameters and input this list to the SGD algorithm constructor.
#
# To verify this, check out the printed parameters to learn. When
# finetuning, this list should be long and include all of the model
# parameters. However, when feature extracting this list should be short
# and only include the weights and biases of the reshaped layers.
#
#
#

# In[9]:


# Send the model to GPU
model_ft = model_ft.to(device)

# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model_ft.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            params_to_update.append(param)
            print("\t", name)
else:
    for name, param in model_ft.named_parameters():
        if param.requires_grad:
            print("\t", name)

# Run Training and Validation Step
# --------------------------------
#
# Finally, the last step is to setup the loss for the model, then run the
# training and validation function for the set number of epochs. Notice,
# depending on the number of epochs this step may take a while on a CPU.
# Also, the default learning rate is not optimal for all of the models, so
# to achieve maximum accuracy it would be necessary to tune for each model
# separately.
#
#
#


# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9, weight_decay=0.0001)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=0.0001)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Train and evaluate
model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, device, checkpoint_save_path,
                             num_epoch_to_stop_if_no_better,
                             num_epochs=num_epochs, is_inception=(model_name == "inception"), is_resume=True)

# Comparison with Model Trained from Scratch
# ------------------------------------------
#
# Just for fun, lets see how the model learns if we do not use transfer
# learning. The performance of finetuning vs. feature extracting depends
# largely on the dataset but in general both transfer learning methods
# produce favorable results in terms of training time and overall accuracy
# versus a model trained from scratch.
#
#
#

# In[13]:


# # Initialize the non-pretrained version of the model used for this run
# scratch_model,_ = initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False)
# scratch_model = scratch_model.to(device)
# scratch_optimizer = optim.SGD(scratch_model.parameters(), lr=0.001, momentum=0.9)
# scratch_criterion = nn.CrossEntropyLoss()
# # _,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=num_epochs, is_inception=(model_name=="inception"))
# _,scratch_hist = train_model(scratch_model, dataloaders_dict, scratch_criterion, scratch_optimizer, num_epochs=15, is_inception=(model_name=="inception"))

# # Plot the training curves of validation accuracy vs. number
# #  of training epochs for the transfer learning method and
# #  the model trained from scratch
# ohist = []
# shist = []

# ohist = [h.cpu().numpy() for h in hist]
# shist = [h.cpu().numpy() for h in scratch_hist]

# num_epochs = 15

# plt.title("Validation Accuracy vs. Number of Training Epochs")
# plt.xlabel("Training Epochs")
# plt.ylabel("Validation Accuracy")
# plt.plot(range(1,num_epochs+1),ohist,label="Pretrained")
# plt.plot(range(1,num_epochs+1),shist,label="Scratch")
# plt.ylim((0,1.))
# plt.xticks(np.arange(1, num_epochs+1, 1.0))
# plt.legend()
# plt.show()


# Final Thoughts and Where to Go Next
# -----------------------------------
#
# Try running some of the other models and see how good the accuracy gets.
# Also, notice that feature extracting takes less time because in the
# backward pass we do not have to calculate most of the gradients. There
# are many places to go from here. You could:
#
# -  Run this code with a harder dataset and see some more benefits of
#    transfer learning
# -  Using the methods described here, use transfer learning to update a
#    different model, perhaps in a new domain (i.e. NLP, audio, etc.)
# -  Once you are happy with a model, you can export it as an ONNX model,
#    or trace it using the hybrid frontend for more speed and optimization
#    opportunities.
#
#
#
