from PIL import Image
import numpy as np

def lfp(im0):
    im0_arr = np.array(im0).astype(np.uint8)
    im_tmp = im0_arr[0:128,0:128,:]
    imgs = []
    imgout = None
    for j in range(10)
        for i in range(10):
            tmp = im0_arr[ 128*i:128*(i+1), 128*j:128*(j+1),:]
            imgs.append(tmp)
        imgs = np.array(imgs)
        print(imgs.shape)
        avg = np.mean(imgs, axis=0)
        print(avg.shape)
        imgs = imgs - avg
        imgs = np.clip(imgs, 0, 255)
        #imgs = imgs.reshape([128, 1280, 3],order='A')
        print(imgs[0].shape)
        imgout0 = imgs[0]
        for i in range(9):
            imgout0 = np.concatenate((imgout, imgs[i+1]))
        print(imgout0.shape)
        if imgout is not None:
            imgout = np.concatenate((imgout, imgout0), axis=1)
        else:
            imgout = imgout0
    imgout = 255 - imgout
    im1 = Image.fromarray(imgout.astype('uint8')).convert('RGB')
    im1.show()

if __name__ == "__main__":
    im0 = Image.open(r"./experiments/1/test_as_male.png")
    lfp(im0)
