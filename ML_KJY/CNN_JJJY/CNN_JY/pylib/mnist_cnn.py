import tensorflow as tf
import random
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
from abc import abstractmethod
import mytool
import matplotlib.pyplot as plt
from myplot import MyPlot
from cnn import CNN


class MnistCNN (CNN):
    db = None
    learning_epoch = None #15
    size_of_segment = None #100

    def load_mnist(self):
        return mytool.load_mnist()

    @abstractmethod
    def my_log(self, i, xdata, ydata):
        err = self.sess.run(self.cost_function, feed_dict={self.X: xdata, self.Y: ydata})
        msg = "Step:{}, Error:{:.6f}".format(i, err)
        self.logs.append(msg)

    def learn_mnist(self, epoch, partial):
        self.learning_epoch = epoch
        self.size_of_segment = partial

        self.db = self.load_mnist()
        print(self.db)
        self.learn_with_segment(self.db, self.learning_epoch, self.size_of_segment)



    # MNIST와 같은 데이터를 이용한 학습
    def learn_with_segment(self, db, learning_epoch, partial_size):
        tf.set_random_seed(777)  # for reproducibility

        self.init_network()  # 가상함수

        self.sess = tf.Session()
        # Initialize TensorFlow variables
        self.sess.run(tf.global_variables_initializer())

        print("\nStart learning:")
        # Training cycle
        for epoch in range(learning_epoch):
            err_4_all_data = 0
            number_of_segment = self.get_number_of_segment()  # 가상함수

            # 처음 데이터를 100개를 읽어 최적화함.
            # 그 다음 100개 데이터에 대하여 수행.
            # 이를 모두 550번 수행하면 전체 데이터 55,000개에 대해 1번 수행하게 됨.
            # 아래 for 문장이 한번 모두 실행되면 전체 데이터에 대해 1번 실행(학습)함.
            for i in range(number_of_segment):
                x_data, y_data = self.get_next_segment()  # 가상함수

                # 아래 에러는 일부분(100개)에 대한 것이므로 전체 에러를 구하려면 550으로 나누어주어야 함. 아래에서 수행
                err_4_partial, _= self.sess.run([self.cost_function, self.optimizer], feed_dict={self.X: x_data, self.Y: y_data})
                err_4_all_data += err_4_partial

            import mytool
            mytool.print_dot()
            avg_err = err_4_all_data / number_of_segment #
            self.costs.append(avg_err)

            self.my_log(epoch, x_data, y_data)  # 가상함수

        print("\nDone!\n")

    def get_number_of_segment(self):
        return int(self.db.train.num_examples / self.size_of_segment) #55,000 / 100

    def get_next_segment(self):
        return self.db.train.next_batch(self.size_of_segment)

    def get_image(self, index):
        # Get one and predict
        image = self.db.test.images[index:index+1]
        return image

    def get_label(self, index):
        label = self.db.test.labels[index:index+1]
        return label

    def get_class(self, index):
        label = self.db.test.labels[index:index+1]
        return self.sess.run(tf.arg_max(label, 1))

    def classify(self, mnist_image):
        category = self.sess.run(tf.argmax(self.hypothesis, 1), feed_dict={self.X: mnist_image})
        return category

    def classify_random(self):
        index = mytool.get_random_int(self.db.test.num_examples)

        image = self.get_image(index)
        label = self.get_class(index)

        category = self.classify(image)
        print('Label', label)
        print('Classified', category)

        self.show_image(image)

    def show_image(self, image):
        plt.imshow(image.reshape(28, 28), cmap='Greys', interpolation='nearest')
        plt.show()

    # 테스트 데이터로 평가
    def evaluate(self):
        # Test model
        is_correct = tf.equal(tf.arg_max(self.hypothesis, 1), tf.arg_max(self.Y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        # Test the model using test sets
        result = accuracy.eval(session=self.sess, feed_dict={self.X: self.db.test.images, self.Y: self.db.test.labels})
        #result = self.sess.run(accuracy, feed_dict={self.X: db.test.images, self.Y: db.test.labels})

        print("Recognition rate :", result)

    def show_error(self):
        mp = MyPlot()
        mp.set_labels('Step', 'Error')
        mp.show_list(self.costs)

    def print_error(self):
        for item in self.costs:
           print(item)

    def show_weight(self):
        print('shape=', self.weights)

        if len(self.weights[0]) is 1:
           mp = MyPlot()
           mp.set_labels('Step', 'Weight')
           mp.show_list(self.weights)
        else:
           print('Cannot show the weight! Call print_weight method.')

    def print_weight(self):
        for item in self.weights:
           print(item)

    def show_bias(self):
        if len(self.weights) is 1:
           mp = MyPlot()
           mp.set_labels('Step', 'Bias')
           mp.show_list(self.biases)
        else:
           print('Cannot show the bias! Call print_bias mehtod.')

    def print_bias(self):
        for item in self.biases:
           print(item)

    def print_log(self):
        for item in self.logs:
           print(item)

