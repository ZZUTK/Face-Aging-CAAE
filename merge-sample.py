import os
import imageio


def mergegif(im, gifdir, interval=0.2):
    frames = []
    for image in image_list:
        if image.endswith('.png'):
            frames.append(imageio.imread(image))
    imageio.mimsave(gifdir, frames, 'GIF', duration = interval)


if __name__ == '__main__':
    pngdir=r'./save/samples/'
    pngfiles = os.listdir(pngdir)
    # files.sort(key = lambda x:int(x[:-4]))
    image_list = [pngdir + img for img in pngfiles]
    gifdir = './save/training_samples.gif'
    mergegif(image_list, gifdir)