#2창뤈 행렬을 전치 안뒤 반환하는 클래스입니다.

class Trasposition :
    def __init__(self, matrix):
        self.ret_matrix =[]
        temp_matrix = []
        x = len(matrix)
        y = len(matrix[0])
        for j in range(x):
            for i in range(y):
                temp_matrix.append(matrix[i][j])
            self.ret_matrix.append(temp_matrix)
            temp_matrix=[]

    def ret_mat(self):
        return self.ret_matrix()
