import io
import base64
from gradcam import GradCAM
from gradcam.utils import visualize_cam
from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import random
import numpy as np
from models.resnet50 import resnet50
import matplotlib.pyplot as plt

PATH = "model_converge.pth.tar"


def load_model():
    model = resnet50(num_classes=2, multitask=True, liu=False,
                     chen=False, CAN_TS=False, crossCBAM=True,
                     crosspatialCBAM=False,  choice=True)

    print("==> Load pretrained model")
    model_dict = model.state_dict()
    checkpoint = torch.load(
        PATH,  map_location=torch.device('cpu'))
    print("load whole weights")
    model.load_state_dict(checkpoint)
    return model


def stringToRGB(base64_string):
    if ',' in base64_string:
        x = base64_string.split(',')[1]
    else:
        x = base64_string
    imgdata = base64.b64decode(str(x))
    image = Image.open(io.BytesIO(imgdata))
    image = image.convert('RGB')
    return image


def preprocess(img):
    torch_img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])(img).to('cpu')
    normed_torch_img = transforms.Normalize(
        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
    return torch_img, normed_torch_img


def get_probas(normed_torch_img, model):
    x = normed_torch_img
    output = model(x)
    output0 = output[0]  # DR
    output1 = output[1]  # DME
    output0 = torch.softmax(output0, dim=1)
    output1 = torch.softmax(output1, dim=1)
    DR = output0.cpu().data.numpy()[0].tolist()
    DME = output1.cpu().data.numpy()[0].tolist()
    print("DR*** normal:{}, DR:{}".format(DR[0], DR[1]))  # 0: normal, 1:DR
    # 0: normal, 1:Mild, 2:Severe
    print(
        "DME*** normal:{}, Mild:{}, Severe:{}".format(DME[0], DME[1], DME[2]))
    return DR, DME


def get_heatmap(torch_img, normed_torch_img, model, idx, proba):
    config = dict(model_type='resnet',
                  arch=model, layer_name='layer4')
    gradcam = GradCAM.from_config(**config)
    mask, _ = gradcam(normed_torch_img, idx)
    heatmap = visualize_cam(mask, torch_img, alpha=proba)
    return heatmap


def RGBToString(img):
    output = io.BytesIO()
    img.save(output, format='JPEG')
    img = output.getvalue()
    base64_string = base64.b64encode(img).decode("utf-8")
    return base64_string
