import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from glob import glob
import imageio
import torch.backends.cudnn as cudnn
from modules.vgg import vgg16, vgg16_bn, vgg19, vgg19_bn
from modules.resnet import resnet50, resnet101, resnet18
import matplotlib.cm
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import cv2
from imagenet_index import index2class
from LRP_util import *
import os
import argparse

# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument('--models', type=str, default='resnet50',
                    help='resnet50 or vgg16 or vgg19')
parser.add_argument('--target_layer', type=str, default='layer2',
                    help='target_layer')
parser.add_argument('--target_class', type=int, default=None,
                    help='target_class')
args = parser.parse_args()

# define data loader
###########################################################################################################################
model_arch = args.models

if model_arch == 'vgg16':
    model = vgg16_bn(pretrained=True).cuda().eval()  #####
    target_layer = model.features[int(args.target_layer)]
elif model_arch == 'vgg19':
    model = vgg19_bn(pretrained=True).cuda().eval()  #####
    target_layer = model.features[int(args.target_layer)]
elif model_arch == 'resnet50':
    model = resnet50(pretrained=True).cuda().eval() #####
    if args.target_layer == 'layer1':
        target_layer = model.layer1
    elif args.target_layer == 'layer2':
        target_layer = model.layer2
    elif args.target_layer == 'layer3':
        target_layer = model.layer3
    elif args.target_layer == 'layer4':
        target_layer = model.layer4
#######################################################################################################################

path_s = os.listdir('./picture')

for path in path_s:
    img_path_long = './picture/{}'.format(path)
    img = cv2.imread(img_path_long,1)
    img_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_show = cv2.resize(img_show,(224,224))
    img = np.float32(cv2.resize(img, (224,224)))/255

    in_tensor = preprocess_image(img).cuda()
    R_CAM, output = model(in_tensor, args.target_layer, [args.target_class])

    if args.target_class == None:
        maxindex = np.argmax(output.data.cpu().numpy())
    else:
        maxindex = args.target_class

    print(index2class[maxindex])
    save_path = './results/{}_{}'.format(index2class[maxindex][:10], img_path_long.split('/')[-1])

    R_CAM = tensor2image(R_CAM)

    fig = plt.figure(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.01)

    plt.subplot(2, 2, 1)
    plt.imshow(img_show)
    plt.title('Original')
    plt.axis('off')

    plt.subplot(2, 2, 2 + 1)
    plt.imshow(img_show)
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow((R_CAM),cmap='seismic')
    plt.imshow(img_show, alpha=.5)
    plt.title('Relevance_CAM', fontsize=15)
    plt.axis('off')

    plt.subplot(2, 2, 2 + 2)
    plt.imshow(img_show*threshold(R_CAM)[...,np.newaxis])
    plt.title('Relevance_CAM', fontsize=15)
    plt.axis('off')

    plt.tight_layout()
    plt.draw()
    plt.waitforbuttonpress()
    plt.savefig(save_path)
    plt.clf()
    plt.close()

print('Done')