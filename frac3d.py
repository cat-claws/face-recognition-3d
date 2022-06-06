import torch
import torch.nn as nn
from torchvision import models

class FRAC3D(nn.Module):
	def __init__(self, ckpt_path = None):
		super(FRAC3D, self).__init__()
		resnet_model = models.resnet50(num_classes=1200)
		resnet_model.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False) 
		resnet_model.load_state_dict(torch.load('ckpt_path', map_location=torch.device('cpu')))
		self.truncated_resnet = nn.Sequential(*list(resnet_model.children())[:-1])
	def forward(self, x):
		feats = self.truncated_resnet(x)
		return feats.view(feats.size(0), -1)
	
