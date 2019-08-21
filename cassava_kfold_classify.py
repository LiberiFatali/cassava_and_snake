import os
import numpy as np
import pandas as pd
import torch

from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from backbone import initialize_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 5
# model_name = "se_resnext50_32x4d"
model_name = "se_resnext101_32x4d"
input_size = 448
num_fold = 5

test_transforms = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

label_names = ['cbb', 'cbsd', 'cgm', 'cmd', 'healthy']


def predict_raw(loaded_model, de, img):
    image_tensor = test_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_tensor = Variable(image_tensor)
    input_tensor = input_tensor.to(de)
    output = loaded_model(input_tensor)
    return output


# Initialize the models to eval
print('Loading checkpoints...')
# model_name_prefix = 'cassava_se_resnext50_32x4d.pth_'
model_name_prefix = 'nocrop_se_resnext101_32x4d.pth_'
list_model = []
for i in range(num_fold):
    fold_model, _ = initialize_model(model_name, num_classes, True, use_pretrained=True)
    ckp_path = 'pytorch_space/' + model_name_prefix + str(i)
    checkpoint = torch.load(ckp_path)
    fold_model.load_state_dict(checkpoint['model_state_dict'])
    fold_model.eval()
    fold_model.to(device)
    list_model.append(fold_model)

# Classify images
print('Classifying...')
test_dir = 'data/test/0'
# test_dir = ''
filenames = os.listdir(test_dir)
with torch.no_grad():
    # for k, fname in enumerate(filenames[:5]):
    for k, fname in enumerate(filenames):
        filepath = os.path.join(test_dir, fname)
        image = Image.open(filepath)

        list_logit = []
        for fold_model in list_model:
            logit = predict_raw(fold_model, device, image)
            list_logit.append(logit.data.cpu().numpy())
        arr_logit = np.array(list_logit)
        averaged_logit = np.mean(arr_logit, axis=0)
        pred_idx = averaged_logit.argmax()

        label = label_names[pred_idx]
        # print(filepath, label, averaged_logit)
        res_dict = {'Category': label_names[pred_idx], 'Id': fname}
        if k == 0:
            out_frame = pd.DataFrame(res_dict, index=[0])
        else:
            out_frame = out_frame.append(res_dict, ignore_index=True)

        if k % 100 == 0:
            print('\t', k, '/', len(filenames))

    out_frame.to_csv('nocrop_se_resnext101_32x4d_5fold.csv', index=False, sep=',', float_format='%.19f')

print('Done')
