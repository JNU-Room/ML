
import os
from PIL import Image


def load_image(image_path = '/home/voidbluelabtop/Desktop/python/ML/ML_KJY/Park_Area/training-images/', cutpoint = None):
    Names = image_path
    name = image_path
    data_image = []
    labels = []
    prd_data = []

    for i in os.listdir(Names):
        if i in ['0', '1']:
            print(i)
            temp = name + str(i)
            FileList = []
            print(temp)
            print(os.listdir(temp))
            for filename in os.listdir(temp):
                if filename.endswith(".jpg"):
                    FileList.append(os.path.join(name,i, filename))

            for filename in FileList:
                Im = Image.open(filename)
                pixel = Im.load()
                width, height = Im.size
                temp2 = []
                for x in range(0, width):
                    temp1 = []
                    for y in range(0, height):
                        temp1.append(list(pixel[x, y]))

                    temp2.append(temp1)
                data_image.append(temp2)
                if i == '0':
                    labels.append([1,0])
                elif i == '1':
                    labels.append([0,1])
        elif i == '2':
            temp = name + str(i)
            FileList = []
            print(temp)
            print(os.listdir(temp))
            for filename in os.listdir(temp):
                if filename.endswith(".jpg"):
                    FileList.append(os.path.join(name, i, filename))
            for filename in FileList:
                print(filename, "이 리스트로 들어갑니다")
                Im = Image.open(filename)
                pixel = Im.load()
                width, height = Im.size
                temp2 = []
                for x in range(0, width):
                    temp1 = []
                    for y in range(0, height):
                        temp1.append(list(pixel[x, y]))

                    temp2.append(temp1)
                prd_data.append(temp2)
    return data_image, labels, prd_data

