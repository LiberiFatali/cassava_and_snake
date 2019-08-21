import os
import numpy as np
import pandas as pd
import torch

# from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms

from backbone import initialize_model
from edafa import ClassPredictor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 45
model_name = "se_resnext50_32x4d"
# model_name = "se_resnext101_32x4d"
input_size = 448
num_fold = 8

test_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def create_readable_names_for_snake_labels():
    """Create a dict mapping label id to human readable string.
    Returns:
        labels_to_names: dictionary where keys are integers from 0 to 44
        and values are human-readable names.
    """

    # pylint: disable=g-line-too-long
    tf_label_path = 'data/labels.txt'
    list_label_id = open(tf_label_path).readlines()

    name_id_path = 'data/e29091a0-37cb-4cb8-a01e-cde5e90fb8a5_class_id_maapping.csv'
    list_name_id = open(name_id_path).readlines()[1:]       # skip header

    cid_to_label = {}
    for s in list_label_id:
        tokens = s.strip().split(':')
        label = int(tokens[0])
        cid = tokens[1]
        cid_to_label[cid] = label

    label_to_name = {}
    for s in list_name_id:
        tokens = s.strip().split(',')
        name = tokens[0]
        cid = '-'.join(['class', tokens[1]])
        label = cid_to_label[cid]
        label_to_name[label] = name

    return label_to_name


def predict_raw(loaded_model, de, img):
    image_tensor = test_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_tensor = Variable(image_tensor)
    input_tensor = input_tensor.to(de)
    output = loaded_model(input_tensor)
    return output


class TtaPredictor(ClassPredictor):
    def __init__(self, trained_model, pipeline, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = trained_model
        self.pipe = pipeline

    def predict_patches(self, patches):
        preds = []
        for i in range(patches.shape[0]):
            processed = self.pipe(patches[i])
            processed = processed.unsqueeze(0)
            processed = Variable(processed)
            processed = processed.to(device)
            pred = self.model(processed)
            preds.append(pred.data.cpu().numpy())
        return np.array(preds)


# use orignal image and flipped Left-Right images
# use arithmetic mean for averaging
tta_conf = '{"augs":["NO", "FLIP_LR", "FLIP_UD"], "mean":"ARITH", "bits":8}'

label_names = create_readable_names_for_snake_labels()
print(label_names)


# Initialize the models to eval
print('Loading checkpoints...')
model_name_prefix = 'centercrop_se_resnext50_32x4d.pth_'
# list_model = []
list_predictor = []
for i in range(num_fold):
    fold_model, _ = initialize_model(model_name, num_classes, True, use_pretrained=True)
    ckp_path = 'snake_trained/' + model_name_prefix + str(i)
    checkpoint = torch.load(ckp_path)
    fold_model.load_state_dict(checkpoint['model_state_dict'])
    fold_model.eval()
    fold_model.to(device)
    # list_model.append(fold_model)

    # for tta
    fold_predictor = TtaPredictor(fold_model, test_transforms, tta_conf)
    list_predictor.append(fold_predictor)

# Classify images
print('Classifying...')
test_dir = 'data/round1'
# test_dir = ''
print('test_dir: ', test_dir)
filenames = os.listdir(test_dir)
out_frame = None
with torch.no_grad():
    # for k, fname in enumerate(filenames[:1]):
    for k, fname in enumerate(filenames):
        filepath = os.path.join(test_dir, fname)
        try:
            # image = Image.open(filepath)
            image = plt.imread(filepath)
        except:
            image = None

        if image is not None:
            list_logit = []
            # for fold_model in list_model:
            #     logit = predict_raw(fold_model, device, image)
            for fold_predictor in list_predictor:
                logit = fold_predictor.predict_images([image])
                list_logit.append(logit)
            arr_logit = np.array(list_logit)
            averaged_logit = np.mean(arr_logit, axis=0)
            # averaged_logit = averaged_logit[0, 0:]
            averaged_logit = averaged_logit.flatten()
            # print('arr_logit: ', arr_logit)
            res_dict = {label_names[i]: p for i, p in enumerate(averaged_logit)}
        else:
            res_dict = {n: 0 for _, n in label_names.items()}

        # print('res_dict: ', res_dict)

        if out_frame is None:
            out_frame = pd.DataFrame(res_dict, index=[0])
            out_frame = out_frame.reindex(sorted(out_frame.columns), axis=1)
            out_frame.insert(loc=0, column='filename', value=[fname])
        else:
            res_dict['filename'] = fname
            out_frame = out_frame.append(res_dict, ignore_index=True)

        if k % 100 == 0:
            print('\t', k, '/', len(filenames))

    out_frame.to_csv('centercrop_se_resnext50_32x4d_8fold.csv', index=False, sep=',', float_format='%.19f')

print('Done')
