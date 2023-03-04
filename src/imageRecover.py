# import ext libs
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import scipy
import sklearn
from scipy.signal import medfilt2d
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as MSE
prepath = os.path.abspath('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/figs')


def imgRead(fileName):
    """
    load the input image into a matrix
    :param fileName: name of the input file
    :return: a matrix of the input image
    Examples: imgIn = imgRead('lena.bmp')
    """
    imgIn = plt.imread(fileName)
    return imgIn


def imgShow(imgOut, fname = ''):
    """
    show the image saved in a matrix
    :param imgOut: a matrix containing the image to show
    :return: None
    """
    # imgOut = np.uint8(imgOut)
    plt.imshow(imgOut)
    plt.title(fname)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar()
    plt.show()
    # plt.savefig(prepath + fname + '.png')
    plt.clf()


def imgBlock(imgOut, K, startRow, startCol):
    crop = []
    if(startRow + K > len(imgOut)):
        startRow = len(imgOut) - K
    if (startCol + K > len(imgOut[0])):
        startCol = len(imgOut[0]) - K
    for i in range(startRow, startRow + K):
        crop.append([])
        for j in range(startCol, startCol + K):
            crop[i-startRow].append(imgOut[i][j])
    return np.asarray(crop)


def compressSimulate(imgOut, p):
    n = len(imgOut)*len(imgOut[0])
    corruptedPixels = set(random.sample(range(0, n), int(p*n)))
    corruptedImage = []
    pxl = 0
    for i in range(len(imgOut)):
        corruptedImage.append([])
        for j in range(len(imgOut[0])):
            if pxl in corruptedPixels:
                corruptedImage[i].append(float('-inf'))
            else:
                corruptedImage[i].append(imgOut[i][j])
            pxl = pxl + 1
    return np.asarray(corruptedImage)


def generateBasisVector(blkSize, u, v):
    Q = blkSize
    P = blkSize
    alpha = np.sqrt(2 / P)
    beta = np.sqrt(2 / 1)
    vec = []
    if u == 1:
        alpha = np.sqrt(1 / P)
    if v == 1:
        beta = np.sqrt(1 / Q)
    for y in range(1, Q + 1):
        for x in range(1, P + 1):
            purple = ((np.pi * ((2 * x) - 1) * (u - 1)) / (2 * P))
            orange = ((np.pi * ((2 * y) - 1) * (v - 1)) / (2 * Q))
            basis = alpha*beta*np.cos(purple)*np.cos(orange)
            vec.append(basis)
    return np.asarray(vec).reshape((Q*P))


def generateBasisMatrix(blkSize):
    mat = []
    Q = blkSize
    P = blkSize
    for u in range(1, P + 1):
        for v in range(1, Q + 1):
            mat.append(generateBasisVector(blkSize, u, v))
    return np.asarray(mat).T


def deletePoints(mat, compressed):
    rasterized = compressed.reshape(len(mat), 1)
    i = 0
    while i < len(rasterized):
        if(rasterized[i][0] == float('-inf')):
            rasterized = np.delete(rasterized, i, 0)
            mat = np.delete(mat, i, 0)
        else:
            i = i + 1
    return [mat, rasterized]


def LASSO(B, A, D, alpha):
    D = np.ravel(D)
    reg = linear_model.Lasso(alpha=alpha, fit_intercept=True) #turn intercept on, remove first column of basis matrix and fit w/ that one, then add back column and fit
    reg.fit(A, D)
    return reg.predict(B)


def LASSOBench(B, A, D):
    D = np.ravel(D)
    reg = linear_model.LassoCV(alphas=np.exp(np.linspace(np.log(10 ** -6), np.log(10 ** 6), 36)), fit_intercept=True, cv=6)
    reg.fit(A, D)
    score = reg.score(A, D)
    means = []
    for path in reg.mse_path_:
        means.append(np.mean(path))
    return [reg.alpha_, means]


def imgReconstruct(preds, compressed):
    K2 = len(compressed) * len(compressed[0])
    rasterized = compressed.reshape(K2, 1)
    for i in range(len(rasterized)):
        if (rasterized[i][0] == float('-inf')):
            rasterized[i][0] = preds[i]
    return rasterized


def imgRecoverBench(imgIn, blkSize, numSample, alpha, x, y, fname, benchmark=False):
    """
    Recover the input image from a small size samples
    :param imgIn: input image
    :param blkSize: block size
    :param numSample: how many samples in each block
    :return: recovered image
    """
    cropped = imgIn
    p = float(((blkSize**2)-numSample)/(blkSize**2))
    c = compressSimulate(cropped, p)
    if benchmark==False:
        plt.title("Corrupted Image, S = " + str(numSample))
        imgShow(c, fname + str(numSample) + 'Pixels' + "x" + str(x) + "y" + str(y))
    B = generateBasisMatrix(c)
    A, D = deletePoints(B, c)
    L = LASSO(B, A, D, alpha)
    reCon = imgReconstruct(L, c)
    mse = MSE(cropped.reshape(blkSize**2), reCon)
    if benchmark == False:
        plt.title("Optimal Regularization Reconstruction, S = " + str(numSample))
        imgShow(reCon.reshape(blkSize, blkSize), fname + str(numSample) + 'PixelsReCon' + "x" + str(x) + "y" + str(y))
    return mse


def stitch(imgIn, K, y, x):
    imgOut = np.zeros((y, x))
    for i in range(y):
        for j in range(x):
            imgOut[i][j] = imgIn[(j//K) + ((x//K)*(i//K))][i % K][j % K]
    return np.asarray(imgOut)


def imgRecover(imgIn, blkSize, numSample, alphas, fname):
    """
    Recover the input image from a small size samples
    :param imgIn: input image
    :param blkSize: block size
    :param numSample: how many samples in each block
    :return: recovered image
    """
    p = float(((blkSize ** 2) - numSample) / (blkSize ** 2))
    B = generateBasisMatrix(blkSize)
    i = len(imgIn)
    j = len(imgIn[0])
    fullImg = []
    compressedImg = []
    for y in range(i//blkSize):
        for x in range(j//blkSize):
            cropped = imgBlock(imgIn, blkSize, blkSize*y, blkSize*x)
            c = compressSimulate(cropped, p)
            test = np.asarray(c)
            compressedImg.append(c.reshape(blkSize, blkSize))
            A, D = deletePoints(B, c)
            L = LASSO(B, A, D, alphas[y][x])
            reCon = imgReconstruct(L, c)
            plt.title("Optimal Regularization Reconstruction, S = " + str(numSample))
            fullImg.append(reCon.reshape(blkSize, blkSize))
            #print(test)
    stitched = stitch(fullImg, blkSize, i, j)
    # compStitched = stitch(compressedImg, blkSize, i, j)
    # imgShow(compStitched)
    med = medfilt2d(stitched)
    rasterRecon = np.reshape(stitched, (-1, 1))
    rasterFilt = np.reshape(med, (-1, 1))
    rasterOG = np.reshape(imgIn, (-1, 1))
    normMSE = MSE(rasterOG, rasterRecon)
    filtMSE = MSE(rasterOG, rasterFilt)
    imgShow(stitched, fname + " Reconstructed, S = " + str(numSample) + ", MSE = " + str(round(normMSE, 3)))
    imgShow(med, fname + " Reconstructed, S = " + str(numSample) + ", MSE = " + str(round(filtMSE, 3)) + ', Median Filtering')
    return stitched, med, normMSE, filtMSE


if __name__ == '__main__':
    boat = imgRead('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/fishing_boat.bmp')
    S = [50, 40, 30, 20, 10]
    K = 8
    alphas = []
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

    for i in range(len(S)):
        imgRecover(boat, 8, S[i], alphas[i], 'Boat')

    nature = imgRead('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/nature.bmp')
    natureAlphas = []
    S2 = [150, 100, 50, 30, 10]
    K2 = 16
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

    for i in range(len(S2)):
        imgRecover(nature, 16, S2[i], natureAlphas[i], 'Nature')