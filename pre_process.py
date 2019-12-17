import cv2 as cv
from  os.path import join
import os
from tqdm import tqdm

# dataset = ['facades', 'cityscapes', 'maps', 'edges2shoes', 'edges2handbags']
dataset = ['facades', 'cityscapes', 'maps']
root = 'data'


def split_image(root, tag):
    path = join(root, tag)
    print(path)
    file_list = os.listdir(path)
    for file in tqdm(file_list):
        full_path = join(path, file)
        img = cv.imread(full_path)
        split = img.shape[1] // 2
        imagea = img[:, :split, :]
        imageb = img[:, split:, :]
        cv.imwrite(path + 'a/' + file, imagea)
        cv.imwrite(path + 'b/' + file, imageb)


def make_dir(name):
    os.mkdir(join(root, name, 'train') + 'a')
    os.mkdir(join(root, name, 'train') + 'b')
    os.mkdir(join(root, name, 'val') + 'a')
    os.mkdir(join(root, name, 'val') + 'b')


def process():
    for name in dataset:
        split_image(join(root, name), 'train')
        split_image(join(root, name), 'val')


if __name__ == '__main__':
    # process()
    split_image(join(root, 'facades'), 'test')
