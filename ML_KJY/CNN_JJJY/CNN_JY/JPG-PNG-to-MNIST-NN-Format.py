'''
Created by gskielian
https://github.com/gskielian/JPG-PNG-to-MNIST-NN-Format
'''



import os
from PIL import Image
from array import *
from random import shuffle

# Load from and save to
Names = [['/home/voidbluelabtop/Desktop/python/ML/ML_KJY/Park_Area/training-images', 'train'], ['/home/voidbluelabtop/Desktop/python/ML/ML_KJY/Park_Area/test-images', 'test']]

for name in Names:

    data_image = []
    data_label = array('B')

    FileList = []
    print(os.listdir(name[0][:]))
    for filename in os.listdir(name[0]):
        if filename.endswith(".png"):
            FileList.append(os.path.join(name[0], filename))

    shuffle(FileList)  # Usefull for further segmenting the validation set
    print(FileList)
    for filename in FileList:


        Im = Image.open(filename)
        print(Im)
        pixel = Im.load()
        width, height= Im.size
        print (width, height)

        for x in range(0, width-1):
            for y in range(0, height-1):
                data_image.append(list(pixel[x, y]))
        print(data_image)

    hexval = "{0:#0{1}x}".format(len(FileList), 6)  # number of files in HEX

    # header for label array

    header = array('B')
    header.extend([0, 0, 8, 1, 0, 0])
    header.append(int('0x' + hexval[2:][:2], 16))
    header.append(int('0x' + hexval[2:][2:], 16))

    data_label = header + data_label
