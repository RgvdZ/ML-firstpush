
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100

def CreateData(N : int  , M : int , inp : str , out : str  ) -> pd.DataFrame :
    """
    Function to create DataSet based on passed parameters.
        N : number of samples
        M : number of features
        inp : 'd' - Discrete , 'r' - Real
        out : 'd' - Discrete , 'r' - Real
    """
    # Create a DataFrame with N rows and M columns filled with random binary values (0 or 1)
    if inp == 'd':
        # Discrete Input Data
        X = pd.DataFrame({i:pd.Series(np.random.randint(M-1, size = N), dtype="category") for i in range(M)})
        if out == 'd':
            # Discrete Output Data
            y = pd.Series(np.random.randint(2, size = N),  dtype="category")
        else :
            # Real Output Data
            y = pd.Series(np.random.randn(N))
    else:
        # Real Input Data
        X = pd.DataFrame(np.random.randn(N, M-1))
        if out == 'd':
            # Discrete Output Data
            y = pd.Series(np.random.randint(2, size = N),  dtype="category")
        else :
            # Real OUtput Data
            y = pd.Series(np.random.randn(N))

    return X,y

def TimingTrees(N : int, M : int , T : int):
    """
    Function to calculate average time (and std) taken by fit() and predict()
    for different N and M for 4 different cases of DTs
    """

    fit_TimingChartdd = np.zeros(shape=(N,M))
    fit_TimingChartrd = np.zeros(shape=(N,M))
    fit_TimingChartdr = np.zeros(shape=(N,M))
    fit_TimingChartrr = np.zeros(shape=(N,M))

    predict_TimingChartdd = np.zeros(shape=(N,M))
    predict_TimingChartrd = np.zeros(shape=(N,M))
    predict_TimingChartdr = np.zeros(shape=(N,M))
    predict_TimingChartrr = np.zeros(shape=(N,M))

    prog = 0
    tot = (M - 5) * ( N - 5 ) * T

    for t in range(T):
        for n in range(5,N):
            for m in range(5,M):

                prog = prog + 1
                print("{0}/{1}\r".format(prog,tot),end=" ")

    #-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

                # Running fit and predict for discrete input and discrete output.
                X,y = CreateData(n,m,'d','d')
                tree = DecisionTree(criterion='information_gain')

                tic = time.perf_counter()
                tree.fit(X,y)
                toc = time.perf_counter()
                fit_TimingChartdd[n][m] = fit_TimingChartdd[n][m] + (toc - tic)

                tic = time.perf_counter()
                tree.predict(X)
                toc = time.perf_counter()
                predict_TimingChartdd[n][m] = predict_TimingChartdd[n][m] + (toc - tic)

    #-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

                # Running fit and predict for discrete input and real output.
                X,y = CreateData(n,m,'d','r')
                tree = DecisionTree(criterion='information_gain')

                tic = time.perf_counter()
                tree.fit(X,y)
                toc = time.perf_counter()
                fit_TimingChartdr[n][m] = fit_TimingChartdr[n][m] + (toc - tic)

                tic = time.perf_counter()
                tree.predict(X)
                toc = time.perf_counter()
                predict_TimingChartdr[n][m] = predict_TimingChartdr[n][m] + (toc - tic)

    #-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

                # Running fit and predict for real input and discrete output.
                X,y = CreateData(n,m,'r','d')
                tree = DecisionTree(criterion='information_gain')

                tic = time.perf_counter()
                tree.fit(X,y)
                toc = time.perf_counter()
                fit_TimingChartrd[n][m] = fit_TimingChartrd[n][m] + (toc - tic)

                tic = time.perf_counter()
                tree.predict(X)
                toc = time.perf_counter()
                predict_TimingChartrd[n][m] = predict_TimingChartrd[n][m] + (toc - tic)

    #-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

                # Running fit and predict for real input and real output.
                X,y = CreateData(n,m,'r','r')
                tree = DecisionTree(criterion='information_gain')

                tic = time.perf_counter()
                tree.fit(X,y)
                toc = time.perf_counter()
                fit_TimingChartrr[n][m] = fit_TimingChartrr[n][m] + (toc - tic)

                tic = time.perf_counter()
                tree.predict(X)
                toc = time.perf_counter()
                predict_TimingChartrr[n][m] = predict_TimingChartrr[n][m] +(toc - tic)

    #-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    fit_TimingChartdd = fit_TimingChartdd / T
    fit_TimingChartrd = fit_TimingChartrd / T
    fit_TimingChartdr = fit_TimingChartdr / T
    fit_TimingChartrr = fit_TimingChartrr / T

    predict_TimingChartdd = predict_TimingChartdd / T
    predict_TimingChartrd = predict_TimingChartrd / T
    predict_TimingChartdr = predict_TimingChartdr / T
    predict_TimingChartrr = predict_TimingChartrr / T

    #-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    PlotTimeChart(fit_TimingChartdd,"Avg fit time for Discrete Input and Discrete Output")
    PlotTimeChart(fit_TimingChartrd,"Avg fit time for Real Input and Discrete Output")
    PlotTimeChart(fit_TimingChartdr,"Avg fit time for Discrete Input and Real Output")
    PlotTimeChart(fit_TimingChartrr,"Avg fit time for Real Input and Real Output")

    #-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

    PlotTimeChart(predict_TimingChartdd,"Avg predict time for Discrete Input and Discrete Output")
    PlotTimeChart(predict_TimingChartrd,"Avg predict time for Real Input and Discrete Output")
    PlotTimeChart(predict_TimingChartdr,"Avg predict time for Discrete Input and Real Output")
    PlotTimeChart(predict_TimingChartrr,"Avg predict time for Real Input and Real Output")

    #-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=


def PlotTimeChart(data : np.array , label ):

    # Determining shape of the data to plot
    n,m = data.shape
    x = range(n)
    y = range(m)

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(x, y)  # plot_surface expects x and y data to be 2D
    ha.plot_surface(X, Y, data)
    ha.set_title(label)
    ha.set_xlabel("Number of Samples")
    ha.set_ylabel("Number of Features")
    name = label +".png"
    plt.savefig(name)

if __name__ == '__main__':
    TimingTrees(25,25,10)

# Learn DTs
# ...
#
# Function to calculate average time (and std) taken by fit() and predict() for different N and P for 4 different cases of DTs
# ...
# Function to plot the results
# ..
# Function to create fake data (take inspiration from usage.py)
# ...
# ..other functions
