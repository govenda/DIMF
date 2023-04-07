import os
from skimage import io
from skimage.color import rgb2gray
from model.MattingModel import EdgeGenerator,Pretrained_Vgg
import torch
import PIL.Image
import time
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

def visualize_single_feature_map(feature,name):

    # plt.imshow(feature,cmap='gray')
    # plt.savefig('result_dm/'+name+".png",bbox_inches ='tight',dpi=500) # 保存图像到本地
    # plt.axis('off')
    # plt.show()
    # 图片分辨率 = figsize*dpi 代码为512*512
    plt.rcParams['figure.figsize'] = (10.24, 10.24)
    plt.rcParams['savefig.dpi'] = 50
    # 去除白框
    plt.axis('off')
    plt.margins(0, 0)
    plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # 保存图片，cmap为调整配色方案
    plt.imshow(feature, cmap=plt.cm.gray)
    plt.savefig('result_dm/'+name+".png")




def test():
    input_dir="testData"
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4500517361627943], [0.26465333914691797])
    ])
    device="cuda:0"
    pre=Pretrained_Vgg().to(device)
    print(pre)
    net = EdgeGenerator().to(device)
    net.load_state_dict(torch.load("experment_transformer/epoch_52_loss_85.884168.pth",map_location="cpu"))
    net.eval()
    t1=time.time()

    for image_name in range(30):
            img1_ori=cv2.imread(os.path.join(input_dir, str(image_name+1) + "-A.jpg"))
            img2_ori = cv2.imread(os.path.join(input_dir, str(image_name + 1) + "-B.jpg"))
            img1 = io.imread(os.path.join(input_dir, str(image_name+1) + "-A.jpg"))
            img2 = io.imread(os.path.join(input_dir, str(image_name+1) + "-B.jpg"))
            img1_gray=rgb2gray(img1)
            img2_gray = rgb2gray(img2)
            ndim = img1.ndim
            #To PIL
            img1_pil = PIL.Image.fromarray(img1)
            img2_pil = PIL.Image.fromarray(img2)
            img1_gray_pil = PIL.Image.fromarray(img1_gray)
            img2_gray_pil = PIL.Image.fromarray(img2_gray)
            #To Tensor
            img1_tensor = data_transforms(img1_pil).unsqueeze(0).to(device)
            img2_tensor = data_transforms(img2_pil).unsqueeze(0).to(device)
            img1_gray_tensor = data_transforms(img1_gray_pil).unsqueeze(0).to(device)
            img2_gray_tensor = data_transforms(img2_gray_pil).unsqueeze(0).to(device)
            dm_m,dm_sf,f1_sf,f2_sf = pre(img1_tensor, img2_tensor)
            f1_sf=f1_sf.squeeze().detach().cpu().numpy()
            f2_sf=f2_sf.squeeze().detach().cpu().numpy()
            dm_sf_save=dm_sf.squeeze().detach().cpu().numpy()
            dm_m_save = dm_m.squeeze().detach().cpu().numpy()
            dm_m = torch.cat([(img1_gray_tensor+img2_gray_tensor)/2, dm_m], dim=1)
            dm = net(dm_m)
            # To binary
            dm = torch.sign(dm - 0.5) / 2 + 0.5
            dm=dm.squeeze().detach().cpu().numpy().astype(np.int)
            # Tensor To Numpy
            if ndim == 3:
                dm = np.expand_dims(dm, axis=2)
            temp_fused = img1_ori * dm + (1 - dm) * img2_ori
            visualize_single_feature_map(f1_sf, 'f1_sf/'+str(image_name+1))
            visualize_single_feature_map(f2_sf, 'f2_sf/'+str(image_name + 1))
            cv2.imwrite(os.path.join("result_dm/dm_sf", str(image_name+1) + ".png"),dm_sf_save*255)
            cv2.imwrite(os.path.join("result_dm/trimap", str(image_name+1) + ".png"),dm_m_save*255)
            cv2.imwrite(os.path.join("result/ssim", str(image_name+1) + ".png"),temp_fused)
            cv2.imwrite(os.path.join("result_dm/fm/ssim", str(image_name + 1) + ".png"), dm*255)
    print(time.time()-t1)


if __name__ == '__main__':
    test()