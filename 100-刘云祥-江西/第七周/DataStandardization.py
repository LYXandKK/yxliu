import numpy as np
import matplotlib.pyplot as plt

def Standardization1(x):
     Xnum = np.zeros([len(x), 1])
     for i in range(len(x)):
        Xnum[i] = (x[i]-min(x))/(max(x) - min(x))
     return Xnum

def Standardization2(x):
    Xnum = np.zeros([len(x), 1])
    for i in range(len(x)):
        Xnum[i] = (x[i]-np.mean(x))/(max(x) - min(x))
    return Xnum

def z_score(x):
    SumValue = 0
    Xnum = np.zeros([len(x), 1])
    for i in range(len(x)):
        a = np.mean(x)
        SumValue = SumValue + (x[i]-np.mean(x))*(x[i]-np.mean(x))
    StdValue = SumValue/len(x)
    for i in range(len(x)):
        Xnum[i] = (x[i] - np.mean(x))/StdValue
    return Xnum

if __name__ == '__main__':
    TestData = [12, 38, 6, 29, 90, 139, 0.7, 42, 58, 68, 79, 88, 199, 189, 209, -186]
    ResultData = Standardization1(TestData)
    ResultData1 = Standardization2(TestData)
    ResultData2 = z_score(TestData)
    print(ResultData)
    print(ResultData1)
    print(ResultData2)
    plt.plot(ResultData2)
    plt.show()
