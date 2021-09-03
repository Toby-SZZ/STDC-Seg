import cv2
import numpy as np

from models.model_stages import BiSeNet
from PIL import Image
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import copy


model = BiSeNet(backbone="STDCNet813", n_classes=19, pretrain_model='./checkpoints/STDCNet813M_73.91.tar',
    use_boundary_2=False, use_boundary_4=False, use_boundary_8=True,
    use_boundary_16=False, use_conv_last=False)

state_dict = torch.load('./checkpoints/train_STDC1-Seg/pths/model_final.pth', map_location='cpu')
model.load_state_dict(state_dict)
model.eval()

input_image = Image.open("/home/toby/prepared_isfl_generated_train_data/crop20/640*480_obstacle_mode_2d_19_ipm_camvid_mono/test/1548281479.369077471.png")
# input_image = Image.open("/home/toby/STDC-Seg/data/leftImg8bit2/val/img005391.png")

input_image = input_image.convert('RGB')

old_img = copy.deepcopy(input_image)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(input_image)
print(
    input_tensor.shape
)

input_batch = input_tensor.unsqueeze(0)

# move the input and model to GPU for speed if available

if torch.cuda.is_available():
    input_batch = input_batch.to('cuda')
    model.to('cuda')

with torch.no_grad():
    output = model(input_batch)[0]

pr = torch.squeeze(output)
print(pr.shape)
output_predictions = pr.argmax(0)

# create a color pallette, selecting a color for each class
# palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
# colors = torch.as_tensor([i for i in range(19)])[:, None] * palette
# colors = (colors % 255).numpy().astype("uint8")
# print (colors.shape)
colors =np.array ([
    [0,0,0],
    [0,0,0],
    [51,0,255],
    [0,255,0],
    [255,0,0],
    [255,255,0],
    [255,153,0],
    [153,102,0],
    [0,102,255],
    [0,102,0],
    [102,51,102],
    [153,255,51],
    [153,0,102],
    [51,51,0],
    [0,204,153],
    [153,102,255],
    [204,153,153],
    [0,102,102],
    [51,204,153]
], dtype='uint8')



# plot the semantic segmentation predictions of 21 classes in each color
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
r.putpalette(colors)
r =r.convert('RGB')

image = Image.blend(old_img,r,0.5)

# plt.imshow(image)
# plt.show()
image.save('./test/test.png')






