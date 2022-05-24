# face-recognition-3d

## Acknowledgement
[FR3DNet的matconv模型转成pytorch模型](https://blog.csdn.net/weixin_43689247/article/details/95613008)

pretrained [checkpoint](https://drive.google.com/file/d/1nLWh9pNB7KZGU3O3Yy_xXnnUy-NGENoB/view?usp=sharing) in PyTorch

## Usage
```python
import torch
from fr3dnet import vgg_face

model = vgg_face()
model.load_state_dict(torch.load('/content/drive/MyDrive/Colab Notebooks/Face-Data/fr3dnet.pth'))
```
