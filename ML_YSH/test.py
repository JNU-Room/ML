from SH_pylib.LinearRegression import LinearRegression
from SH_pylib.LogisticClassification import LogisticClassification

lr = LinearRegression()

lr.linear_regression('train_LR.txt')
lr.what_is_it([15])

lc = LogisticClassification()

lc.logistic_classification('train_LC.txt')
lc.what_is_it([3,2])