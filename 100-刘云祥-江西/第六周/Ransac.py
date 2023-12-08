import numpy
import numpy as np
import scipy as sp
import scipy.linalg as sl





def Ransac(data, model, n, k, t, d, debug=False, return_all=False):
    '''
    输入：
    :param data: 样本数
    :param model: 假设应用的模型，需提前确定
    :param n: 生成模型所需的最小样本数量
    :param k: 最大迭代次数
    :param t: 阈值：作为判断点满足模型的条件
    :param d:拟合较好时，需要的样本点最小的个数
    输出：
    bestfit-最优拟合解（返回nil，如果未找到）
    '''
    iterations = 0
    bestfit = None
    besterr = np.inf
    best_inlier_idxs = None
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybe_inliers = data[maybe_idxs, :]  # 得到size（maybe_idxs)行数据（Xi，Yi）
        test_points = data[test_idxs]
        maybemodel = model.fit(maybe_inliers)
        test_err = model.get_error(test_points, maybemodel)  # 计算误差，平方和最小
        print('test_err = ', test_err < t)
        also_idxs = test_idxs[test_err < t]
        print('also_idxs = ', also_idxs)
        also_inliers = data[also_idxs, :]
        if debug:
            print('test_err.min)()', test_err.min())
            print('test_err.max()', test_err.max())
            print('numpy.mean(test_err)', numpy.mean(test_err))
            print('iteration %d:len(alsoinliers) = %d' % (iterations, len(also_inliers)))
        print('d = ', d)
        if (len(also_inliers) > d):
            betterdata = np.concatenate((maybe_inliers, also_inliers))  # 样本连接
            bettermodel = model.fit(betterdata)
            better_errs = model.get_error(betterdata, bettermodel)
            thiserr = np.mean(better_errs)  # 平均误差作为新误差
            if thiserr < besterr:
                bestfit = bettermodel
                besterr = thiserr
                best_inlier_idxs = np.concatenate((maybe_idxs, also_idxs))  # 更新内点，将新点加入
        iterations += 1
    if bestfit is None:
        raise ValueError("didn't meet fit acceptance criteria")
    if return_all:
        return bestfit, {'inliers': best_inlier_idxs}
    else:
        return bestfit


def random_partition(n, n_data):
    all_idxs = np.arange(n_data)
    np.random.shuffle(all_idxs)  # 打乱下标索引
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2

class LinearLeastSquareModel:
    # 最小二乘求线性解，用于RANSAC的输入模型
    def __init__(self, input_columns, output_columns, debug=False):
        self.input_columns = input_columns
        self.output_columns = output_columns
        self.debug = debug

    def fit(self, data):
        # np.vstack按照垂直方向（行顺序）堆叠数值构成新数组
        A = np.vstack([data[:, i] for i in self.input_columns]).T  # 第一列Xi-行Xi
        B = np.vstack([data[:, i] for i in self.output_columns]).T  # 第二列Yi-行Yi
        x, resids, rank, s = sl.lstsq(A, B)  # residues:残差和
        return x  # 返回最小平方和向量

    def get_error(self, data, model):
        A = np.vstack([data[:, i] for i in self.input_columns]).T
        B = np.vstack([data[:, i] for i in self.output_columns]).T
        B_fit = sp.dot(A, model)  # B_fit=model.k*A+model.b
        err_per_point = np.sum((B - B_fit) ** 2, axis=1)
        return err_per_point

def test():
    n_samples = 500  # 样本个数
    n_inputs = 1  # 输入变量个数
    n_outputs = 1  # 输出变量个数
    A_exact = 20 * np.random.random((n_samples, n_inputs))  # 随机生成0-20的500个数据：行向量
    perfect_fit = 60 * np.random.normal(size=(n_inputs, n_outputs))  # 随机生成斜率
    B_exact = sp.dot(A_exact, perfect_fit)  # y = x * k

    # 加入高斯噪声
    A_noisy = A_exact + np.random.normal(size=A_exact.shape)
    B_noisy = B_exact + np.random.normal(size=B_exact.shape)

    if 1:
        # 添加“局外点”
        n_outliers = 100
        all_idxs = np.arange(A_noisy.shape[0])  # 索引0-499
        np.random.shuffle(all_idxs)  # all_idxs打乱
        outlier_idxs = all_idxs[:n_outliers]  # 100个0-500的随机局外点
        A_noisy[outlier_idxs] = 20 * np.random.random((n_outliers, n_inputs))
        B_noisy[outlier_idxs] = 50 * np.random.normal(size=(n_outliers, n_outputs))

        all_data = np.hstack((A_noisy, B_noisy))
        input_columns = range(n_inputs)
        output_columns = [n_inputs + i for i in range(n_outputs)]
        debug = False
        model = LinearLeastSquareModel(input_columns, output_columns, debug=debug)  # 生成已知模型
        linear_fit, resids, rank, s = sp.linalg.lstsq(all_data[:, input_columns], all_data[:, output_columns])

        ransac_fit, ransac_data = Ransac(all_data, model, 50, 1000, 7e3, 300, debug=debug, return_all=True)

    if 1:
        import pylab

        sort_idxs = np.argsort(A_exact[:, 0])
        A_col0_sorted = A_exact[sort_idxs]  # 秩为2的数组

        if 1:
            pylab.plot(A_noisy[:, 0], B_noisy[:, 0], 'k.', label='data')  # 散点图
            pylab.plot(A_noisy[ransac_data['inliers'], 0], B_noisy[ransac_data['inliers'], 0], 'bx',
                       label="RANSAC data")
        else:
            pylab.plot(A_noisy[non_outlier_idxs, 0], B_noisy[non_outlier_idxs, 0], 'k.', label='noisy data')
            pylab.plot(A_noisy[outlier_idxs, 0], B_noisy[outlier_idxs, 0], 'r.', label='outlier data')

        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, ransac_fit)[:, 0],
                   label='RANSAC fit')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, perfect_fit)[:, 0],
                   label='exact system')
        pylab.plot(A_col0_sorted[:, 0],
                   np.dot(A_col0_sorted, linear_fit)[:, 0],
                   label='linear fit')
        pylab.legend()
        pylab.show()

if __name__ == '__main__':
  test()
