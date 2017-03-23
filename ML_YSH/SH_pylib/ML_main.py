# 파일들 import

class MachineLearning:

    # training data set 설정

    # 학습 선택
    # 1: One-variable Linear Regression / 2: Multi-variable Linear Regression
    # 3: Logistic Classification / 4: Multinomial Classification
    def select_learning(self, choice):
        if choice == 1: # 1: One-variable Linear Regression
            print ("One-variable Linear Regression")
        elif choice == 2: # 2: Multi-variable Linear Regression
            print ("Multi-variable Linear Regression")
        elif choice == 3: # 3: Logistic Classification
            print ("Logistic Classification")
        elif choice == 4: # 4: Multinomial Classification
            print ("Multinomial Classification")
        else:
            print ("1: One-variable Linear Regression / 2: Multi-variable Linear Regression / 3: Logistic Classification / 4: Multinomial Classification")
            print ("중에서 입력해주세요.")

    # 학습 결과 출력

    # 예측

# main
gildong = MachineLearning()
gildong.select_learning(1)