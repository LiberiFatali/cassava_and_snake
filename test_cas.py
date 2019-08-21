import os
import pandas as pd
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms

from backbone import initialize_model

# inference parameters
model_name = "resnet-152"
feature_extract = False
num_classes = 5
checkpoint_save_path = './pytorch_space/resnet152_cas.pth'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Initialize the model for this run
model, input_size = initialize_model(model_name,
                                     num_classes,
                                     feature_extract,
                                     use_pretrained=True)
checkpoint = torch.load(checkpoint_save_path)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
model.to(device)

test_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

label_names = ['cbb', 'cbsd', 'cgm', 'cmd', 'healthy']


def predict_image(image):
    image_tensor = test_transforms(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_tensor = Variable(image_tensor)
    input_tensor = input_tensor.to(device)
    output = model(input_tensor)
    index = output.data.cpu().numpy().argmax()
    return index


test_path = 'data/test/0'
filenames = os.listdir(test_path)
with torch.no_grad():
    # Iterate over data.
    # for i, fname in enumerate(filenames[:5]):
    for i, fname in enumerate(filenames):
        filepath = os.path.join(test_path, fname)
        img = Image.open(filepath)
        pred_idx = predict_image(img)
        label = label_names[pred_idx]
        # print(filepath, label)
        res_dict = {'Category': label_names[pred_idx], 'Id': fname}
        if i == 0:
            out_frame = pd.DataFrame(res_dict, index=[0])
        else:
            out_frame = out_frame.append(res_dict, ignore_index=True)

    out_frame.to_csv('resnet152_result.csv',
                     index=False,
                     sep=',',
                     float_format='%.19f')
