
import os
from PIL import Image


def load_image(cutpoint, image_path = '/home/voidbluelabtop/Desktop/python/ML/ML_KJY/Park_Area/training-images/'):
    Names = image_path
    name = image_path
    oneimage_set = []
    labels = []
    prd_data = []
    #경로 내에 있는 디렉터리만큼 반복
    for i in os.listdir(Names):
        if i in ['0', '1']:
            print(i)
            temp = name + str(i)
            FileList = []
            for filename in os.listdir(temp):
                if filename.endswith(".jpg"):
                    FileList.append(os.path.join(name,i, filename))
            #디렉터리 내 이미지의 수만큼 반복
            for filename in FileList:
                #이미지를 쪼갠 수 만큼 반복, 이미지를 쪼개는 작업
                for w, point in enumerate(cutpoint):
                    Im = Image.open(filename)
                    pixel = Im.load()
                    width_start = point[0]
                    height_start = point[1]
                    width_end = point[2]
                    height_end = point[3]
                    column = []
                    #잘라진 이미지를 리스트에 넣음
                    for y in range(height_start, height_end):
                        row = []
                        for x in range(width_start, width_end):
                            row.append(list(pixel[x, y]))
                        column.append(row)

                    oneimage_set.append(column)

                f = open(filename[:-4] + '.txt')
                label_text = f.read()
                label= label_text.split(' ')
                labels.append(label)
                f.close()

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

