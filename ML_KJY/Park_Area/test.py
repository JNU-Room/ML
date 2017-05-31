from CNNStudy import CNN
import load_image
import numpy as np

x_data = load_image.load_image()[0]
y_data = load_image.load_image()[1]
x_data = np.array(x_data, dtype=np.float32)

cnn = CNN(load_image,y_data)

xxx = cnn.convolution_layer(x_data,filter_x= 3, filter_y= 3, depth = 3,  num_of_filter= 5, move_right = 1, move_down = 1)
xxx = cnn.convolution_layer(xxx,filter_x= 3, filter_y= 3, depth = 5,  num_of_filter= 10, move_right = 1, move_down = 1)
xxx = cnn.convolution_layer(xxx,filter_x= 3, filter_y= 3, depth = 10,  num_of_filter= 5, move_right = 1, move_down = 1)
xxx = cnn.convolution_layer(xxx,filter_x= 3, filter_y= 3, depth = 5,  num_of_filter= 5, move_right = 1, move_down = 1)
xxx = cnn.final_layer(xxx,len(x_data[0]), len(x_data[0][0]))

cnn.set_cost_function()
cnn.train(x_data = xxx,y_data = y_data, keep_prop = 0.7, step = 200)

prd_data = load_image.load_image()[2]
prd_data =  np.array(prd_data, dtype=np.float32)
cnn.predict(prd_data,4)
