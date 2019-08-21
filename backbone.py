import torch.nn as nn
import pretrainedmodels
from torchvision import models


# Initialize and Reshape the Networks
# Now to the most interesting part. Here is where we handle the reshaping of each network. Note, this is not an
# automatic procedure and is unique to each model. Recall, the final layer of a CNN model, which is often times an
# FC layer, has the same number of nodes as the number of output classes in the dataset. Since all of the models have
# been pretrained on Imagenet, they all have output layers of size 1000, one node for each class. The goal here is to
# reshape the last layer to have the same number of inputs as before, AND to have the same number of outputs as the
# number of classes in the dataset. In the following sections we will discuss how to alter the architecture of each
# model individually. But first, there is one important detail regarding the difference between finetuning and
# feature-extraction.
#
# When feature extracting, we only want to update the parameters of the last layer, or in other words, we only want to
# update the parameters for the layer(s) we are reshaping. Therefore, we do not need to compute the gradients of the
# parameters that we are not changing, so for efficiency we set the .requires_grad attribute to False. This is important
# because by default, this attribute is set to True. Then, when we initialize the new layer and by default the new
# parameters have .requires_grad=True so only the new layer’s parameters will be updated. When we are finetuning we can
# leave all of the .required_grad’s set to the default of True.
#
# Finally, notice that inception_v3 requires the input size to be (299,299), whereas all of the other models expect (224,224).


def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "se_resnext50_32x4d":
        model_ft = pretrainedmodels.se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        # print('model_ft.last_linear.in_features: ', num_ftrs)
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "se_resnext101_32x4d":
        model_ft = pretrainedmodels.se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "PolyNet":
        model_ft = pretrainedmodels.polynet(num_classes=1000, pretrained='imagenet')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "SENet154":
        model_ft = pretrainedmodels.senet154(num_classes=1000, pretrained='imagenet')
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet-18":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet-50":
        """ Resnet50
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        # model_ft.fc = nn.Linear(num_ftrs, num_classes)
        model_ft.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(num_ftrs, num_classes)
        )
        input_size = 224

    elif model_name == "resnet-152":
        """ Resnet152
        """
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size
