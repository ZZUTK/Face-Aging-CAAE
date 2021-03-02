import os
import imageio
import argparse


parser = argparse.ArgumentParser(description='CAAE')
parser.add_argument('--dir', '-d', type=str)

def mergegif(im, gifdir, interval=0.2):
    frames = []
    for image in image_list:
        if image.endswith('.png'):
            frames.append(imageio.imread(image))
    imageio.mimsave(gifdir, frames, 'GIF', duration = interval)


if __name__ == '__main__':
    args = parser.parse_args()
    if args.dir == 'samples':
        pngdir = './save/samples/'
        gifdir = './save/training_samples.gif'
    elif args.dir == 'test':
        pngdir = './save/test/'
        gifdir = './save/test_outputs.gif'
    else:
        print("No such directory >_<")
        exit()
    pngfiles = os.listdir(pngdir)
    # files.sort(key = lambda x:int(x[:-4]))
    image_list = [pngdir + img for img in pngfiles]
    if args.dir == 'test':
        image_list = image_list[0:-1]
        print(len(image_list))
    mergegif(image_list, gifdir)