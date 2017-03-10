#2차원 행렬을 전치 한 뒤 반환하는 클래스

class Trasposition :
    ret_matrix = []
    def __init__(self, matrix):
        temp_matrix = []
        x = len(matrix)
        y = len(matrix[0])
        for j in range(y):
            for i in range(x):
                temp_matrix.append(matrix[i][j])
            self.ret_matrix.append(temp_matrix)
            temp_matrix=[]

    def ret_mat(self):
        return self.ret_matrix
