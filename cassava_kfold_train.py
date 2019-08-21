import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
import pickle
from torchvision import datasets, transforms
from backbone import initialize_model
from helper import train_model
from sklearn.model_selection import StratifiedKFold

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = 'data/train'

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
# model_name = "se_resnext50_32x4d"
model_name = "se_resnext101_32x4d"

num_fold = 5

# Number of classes in the dataset
num_classes = 5

# Batch size for training (change depending on how much memory you have)
batch_size = 10

base_lr = 0.0002

# Number of epochs to train for
num_epochs = 20
num_epoch_to_stop_if_no_better = 50

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

#
input_size = 448

#
checkpoint_save_path = './pytorch_space/nocrop_se_resnext101_32x4d.pth'

# Initialize the model for this run
# model_ft, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
# print(model_ft)

# # Send the model to GPU
# model_ft = model_ft.to(device)
#
# # Gather the parameters to be optimized/updated in this run. If we are
# #  finetuning we will be updating all parameters. However, if we are
# #  doing feature extract method, we will only update the parameters
# #  that we have just initialized, i.e. the parameters with requires_grad
# #  is True.
# params_to_update = model_ft.parameters()
# print("Params to learn:")
# if feature_extract:
#     params_to_update = []
#     for name, param in model_ft.named_parameters():
#         if param.requires_grad:
#             params_to_update.append(param)
#             print("\t", name)
# else:
#     for name, param in model_ft.named_parameters():
#         if param.requires_grad:
#             print("\t", name)


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
        # transforms.RandomResizedCrop(input_size),
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=30),
        # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        # transforms.ColorJitter(brightness=0.4,
        #                        contrast=0.4,
        #                        saturation=0.4,
        #                        hue=0.2),
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

# # Observe that all parameters are being optimized
# # optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
# optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, weight_decay=0.0001)
#
# # Setup the loss fxn
# criterion = nn.CrossEntropyLoss()

print("Initializing Datasets and Dataloaders...")

# Load full set
full_dataset = datasets.ImageFolder(data_dir)
full_label = np.array([s[1] for s in full_dataset])

list_hist = []
list_best_acc = []
kf = StratifiedKFold(n_splits=num_fold, shuffle=True)
splits = kf.split(full_dataset, full_label)
for i, (train_idxs, test_idxs) in enumerate(splits):
    print('============ ', 'Fold ', i, ' ============')

    train_subset = torch.utils.data.Subset(full_dataset, train_idxs)
    val_subset = torch.utils.data.Subset(full_dataset, test_idxs)
    train_fold = CassavaDataset(train_subset, transform=data_transforms['train'])
    val_fold = CassavaDataset(val_subset, transform=data_transforms['val'])
    train_loader = torch.utils.data.DataLoader(train_fold, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_fold, batch_size=batch_size, shuffle=True, num_workers=4)

    # Create training and validation dataloaders
    dataloaders_dict = {
        'train': train_loader,
        'val': val_loader
    }

    # Initialize the model for this fold
    fold_model, _ = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
    fold_model = fold_model.to(device)
    # fold_optimizer = optim.Adam(fold_model.parameters(), lr=0.0001, weight_decay=0.0001)
    fold_optimizer = optim.Adam(fold_model.parameters(), lr=base_lr)
    # Setup the loss fxn
    fold_criterion = nn.CrossEntropyLoss()

    # Run Training and Validation Step
    _, hist, best_acc = train_model(fold_model, dataloaders_dict, fold_criterion, fold_optimizer, device,
                                    checkpoint_save_path,
                                    num_epoch_to_stop_if_no_better, fold_idx=i, num_epochs=num_epochs,
                                    is_inception=(model_name == "inception"), is_resume=False)
    list_hist.append(hist)
    list_best_acc.append(best_acc.cpu().numpy())

print('Best averaged acc of all folds ', np.mean(list_best_acc))

with open('list_hist.pkl', 'wb') as f:
    pickle.dump(list_hist, f)
