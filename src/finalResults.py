import numpy as np
from imageRecover import imgRead
from imageRecover import imgShow
from imageRecover import imgRecover
from sklearn.metrics import mean_squared_error as MSE

if __name__ == '__main__':
    alphas = []
    S = [50, 40, 30, 20, 10]
    # boat = imgRead('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/fishing_boat.bmp')
    # nature = imgRead('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/nature.bmp')
    alphas.append(
        np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/bestAlphasS50.csv', delimiter=','))
    alphas.append(
        np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/bestAlphasS40.csv', delimiter=','))
    alphas.append(
        np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/bestAlphasS30.csv', delimiter=','))
    alphas.append(
        np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/bestAlphasS20.csv', delimiter=','))
    alphas.append(
        np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/bestAlphasS10.csv', delimiter=','))
    # alphas.append(
    #     np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/bestAlphasSLASSO50.csv', delimiter=','))
    # alphas.append(
    #     np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/bestAlphasSLASSO40.csv', delimiter=','))
    # alphas.append(
    #     np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/bestAlphasSLASSO30.csv', delimiter=','))
    # alphas.append(
    #     np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/bestAlphasSLASSO20.csv', delimiter=','))
    # alphas.append(
    # #     np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/bestAlphasSLASSO10.csv', delimiter=','))
    # for i in range(len(S)):
    #     noFilt,filt = imgRecover(boat, 8, S[i], alphas[i], 'Boat')
    #     rasterRecon = np.reshape(noFilt, (-1, 1))
    #     rasterFilt = np.reshape(filt, (-1, 1))
    #     rasterOG = np.reshape(boat, (-1, 1))
    #     print(MSE(rasterOG, rasterRecon))
    #     print(MSE(rasterOG, rasterFilt))

    natureAlphas = []
    natureAlphas.append(
        np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/natureBestAlphasS150.csv', delimiter=','))
    natureAlphas.append(
        np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/natureBestAlphasS100.csv', delimiter=','))
    natureAlphas.append(
        np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/natureBestAlphasS50.csv', delimiter=','))
    natureAlphas.append(
        np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/natureBestAlphasS30.csv', delimiter=','))
    natureAlphas.append(
        np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/natureBestAlphasS10.csv', delimiter=','))
    # natureAlphas.append(
    #     np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/natureBestAlphasSLASSO150.csv', delimiter=','))
    # natureAlphas.append(
    #     np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/natureBestAlphasSLASSO100.csv', delimiter=','))
    # natureAlphas.append(
    #     np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/natureBestAlphasSLASSO50.csv', delimiter=','))
    # natureAlphas.append(
    #     np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/natureBestAlphasSLASSO30.csv', delimiter=','))
    # natureAlphas.append(
    #     np.loadtxt('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/natureBestAlphasSLASSO10.csv', delimiter=','))
    for i in range(len(alphas)):
        imgShow(np.log10(alphas[i]), 'log10(regularization parameter) of Boat by Image Block, S = ' + str(S[i]))
    S = [150, 100, 50, 30, 10]
    for i in range(len(alphas)):
        imgShow(np.log10(natureAlphas[i]), 'log10(regularization parameter) of Nature by Image Block, S = ' + str(S[i]))