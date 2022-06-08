import torch
import math
import torch.nn as nn

class vgg_face(nn.Module):
	def __init__(self,num_classes=1853, data_mode = 'ONE_CHANNEL'):
		super(vgg_face,self).__init__()
		inplace = True
		self.conv1_1 = nn.Conv2d(3,64,kernel_size=(7,7),stride=(1,1),padding=(3,3))
		self.relu1_1 = nn.ReLU(inplace)
		self.conv1_2 = nn.Conv2d(64,64,kernel_size=(7,7),stride=(1,1),padding=(3,3))
		self.relu1_2 = nn.ReLU(inplace)
		self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)

		self.conv2_1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu2_1 = nn.ReLU(inplace)
		self.conv2_2 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu2_2 = nn.ReLU(inplace)
		self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)

		self.conv3_1 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu3_1 = nn.ReLU(inplace)
		self.conv3_2 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu3_2 = nn.ReLU(inplace)
		self.conv3_3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu3_3 = nn.ReLU(inplace)
		self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)

		self.conv4_1 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu4_1 = nn.ReLU(inplace)
		self.conv4_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu4_2 = nn.ReLU(inplace)
		self.conv4_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu4_3 = nn.ReLU(inplace)
		self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False)

		self.conv5_1 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu5_1 = nn.ReLU(inplace)
		self.conv5_2 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu5_2 = nn.ReLU(inplace)
		self.conv5_3 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
		self.relu5_3 = nn.ReLU(inplace)
		self.pool5 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), dilation=(1, 1), ceil_mode=False) 



		self.fc6 = nn.Linear(in_features=512*5*5, out_features=1024, bias=True)
		self.relu6 = nn.ReLU(inplace)
		self.drop6 = nn.Dropout(p=0.5)

		self.fc7 = nn.Linear(in_features=1024, out_features=1024, bias=True)
		self.relu7 = nn.ReLU(inplace)
		self.drop7 = nn.Dropout(p=0.5)
		self.fc8 = nn.Linear(in_features=1024, out_features=num_classes, bias=True)

		self._initialize_weights()
	def forward(self,x, y=None):
		out = self.conv1_1(x)
		x_conv1 = out
		out = self.relu1_1(out)
		out = self.conv1_2(out)
		out = self.relu1_2(out)
		out = self.pool1(out)
		x_pool1 = out

		out = self.conv2_1(out)
		out = self.relu2_1(out)
		out = self.conv2_2(out)
		out = self.relu2_2(out)
		out = self.pool2(out)
		x_pool2 = out

		out = self.conv3_1(out)
		out = self.relu3_1(out)
		out = self.conv3_2(out)
		out = self.relu3_2(out)
		out = self.conv3_3(out)
		out = self.relu3_3(out)
		out = self.pool3(out)
		x_pool3 = out

		out = self.conv4_1(out)
		out = self.relu4_1(out)
		out = self.conv4_2(out)
		out = self.relu4_2(out)
		out = self.conv4_3(out)
		out = self.relu4_3(out)
		out = self.pool4(out)
		x_pool4 = out

		out = self.conv5_1(out)
		out = self.relu5_1(out)
		out = self.conv5_2(out)
		out = self.relu5_2(out)
		out = self.conv5_3(out)
		out = self.relu5_3(out)
		out = self.pool5(out)
		x_pool5 = out

		out = self.fc6(out.view(out.size(0),-1))
		out = self.relu6(out)
		out = self.fc7(out)
		out = self.relu7(out)
		out = self.fc8(out)
		return out

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(0, 0.01)
				m.bias.data.zero_()


class FR3DNET(nn.Module):
	def __init__(self, ckpt_path = None):
		super(FR3DNET, self).__init__()
		vgg = vgg_face()
		vgg.load_state_dict(torch.load(ckpt_path, map_location=torch.device('cpu')))
		self.truncated_vgg = nn.Sequential(*list(vgg.children())[:-1])
	def forward(self, x):
		feats = self.truncated_vgg(x)
		return feats.view(feats.size(0), -1)
