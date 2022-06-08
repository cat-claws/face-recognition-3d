import torch
from fr3dnet import FR3DNET
from led3d import Led3D
from frac3d import FRAC3D

def get_3d_model(checkpoint):
	if 'resnet50' in checkpoint.lower():
		model = FRAC3D(checkpoint) # RGB-D-ResNet50-from-imagenet.pkl
	elif 'led' in checkpoint.lower():
		model = Led3D(checkpoint) # led3d.pth
	elif 'fr3dnet' in checkpoint.lower():
		model = FR3DNET(checkpoint) # fr3dnet.pth
	
	return model.eval()
