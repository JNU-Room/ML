import tensorflow as tf
from pylib.Softmax import Softmax
import numpy as np

max = 4  #단어의 최대 길이
char_list = [0,'q','a','z','w','s','x','e','d','c','r','f','v','t','g','b','y','h','n','u','j','m','i','k','o','l','p']
char_dic = {w : i for i, w in enumerate(char_list)}



word_list = ['my', 'name', 'is', 'kim', 'jae', 'yun', '!','gg', 'han', 'sung']
word_dic = {w : i for i, w in enumerate(word_list)}   #word to code
word_cdic = {i : w for i, w in enumerate(word_list)}  #code to num

x_data = 'mi nema ls kam jie sung mi name id kim jaa sunt my nma is kim jie suhn mi namt us kim jee usnh my name is kim jae suhn nu anse is kim aje thn'
x_data = x_data.split(' ')
x_list = np.zeros([len(x_data),max+1])  #단어들을 숫자로 바꾸기 위한 리스트

y_data = 'my name is kim jae sung my name is kim jae sung my name is kim jae sung my name is kim jae sung my name is kim jae yun my name is kim jae sung'
y_data = y_data.split(' ')
y_word_token = []
y_list = np.zeros([len(y_data),max + 1])
y_label = np.zeros([len(y_data),len(word_dic)]) #단어들을 숫자로 바꾸기 위한 리스트

#학습 시킬 떄 input, output을 숫자형태의 리스트로 바꾸는 과정
for i in range(len(y_data)):
    temp_list = list(y_data[i])
    for j in range(len(y_data[i])):
        y_list[i][j] = char_dic[y_data[i][j]]
    for j in range(len(y_data[i]) + 1, 5):
        y_list[i][j]= 0
    y_list[i][max] = len(y_data[i])

for i in range(len(x_data)):
    print(x_data[i])
    for j in range(len(x_data[i])):
        x_list[i][j] = char_dic[x_data[i][j]]
    for j in range(len(x_data[i]) + 1, 5):
        x_list[i][j] = 0
    x_list[i][max] = len(x_data[i])
print(x_list)

#소프트맥스의 라벨형태로 바꿈
for i in range(len(y_data)):
    y_label[i][word_dic[y_data[i]]] = 1
first_layer = Softmax(x_list, y_label)
first_layer.training(learning_rate=0.005, step=5001, show_training_data=False)
first_layer.predict(x_list)
first_output_pos = first_layer.return_predict_possibility()
first_output_onehot = first_layer.return_predict_onehot()
print (first_output_pos)
# for i in range(len(x_data)):
#     for j in range(len(x_data[i])):
#         first_output[i][j] = char_dic[x_data[i][j]]
#     for j in range(len(x_data[i]) + 1, max+1):
#         first_output[i][j] = 0
#         first_output[i][max] = len(x_data[i])

#두번쨰 레이어의 입력으로 들어각 위한 처리, 첫번쨰 레이어의 output과 번역할 문장의 길이를 파라미터로 받는다.
#나오는 값은 각 단어가 맞을 확률에 앞단어와 뒷단어가 추가된 리스트이다.
def second_input(f_possibility, f_onehot):
    ret_list = []
    ret2_list = []
    for i in range(len(f_possibility)):
        ret_list = list(f_possibility[0])
        if i == 0:
            ret_list.append(-1)
            ret_list.append(f_onehot[i+1])
        elif i == len(f_possibility) - 1:
            ret_list.append(f_onehot[i-1])
            ret_list.append(-1)
        else:
            ret_list.append(f_onehot[i-1])
            ret_list.append(f_onehot[i+1])
        ret2_list.append(ret_list)
    return ret2_list

second_layer = Softmax(second_input(first_output_pos,first_output_onehot), y_label)
second_layer.training(learning_rate=0.1, step=5001, show_training_data=False)
prd = ['mi', 'nema', 'ls', 'aim', 'jie', 'yin']
prd_list = np.zeros([len(prd),max + 1])
for i in range(len(prd)):
    for j in range(len(prd[i])):
        prd_list[i][j] = char_dic[prd[i][j]]
    for j in range(len(x_data[i]) + 1, max+1):
        prd_list[i][j] = 0
    prd_list[i][max] = len(prd[i])
first_layer.predict(prd_list)
second_layer.predict(second_input(first_layer.return_predict_possibility(),first_output_onehot))
print(first_layer.return_predict_possibility())
print(second_layer.return_predict_possibility())
print(second_layer.return_predict_onehot())
print(second_input(first_layer.return_predict_possibility(),first_output_onehot))
text = []
for i in second_layer.return_predict_onehot():
    text.append(word_cdic[i])
print(text)
