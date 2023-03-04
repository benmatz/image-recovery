# import ext libs
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import scipy
import sklearn
from sklearn import linear_model
from sklearn.metrics import mean_squared_error as MSE
from imageRecover import imgRecover, generateBasisMatrix, imgBlock, compressSimulate, deletePoints
from imageRecover import imgRead
from imageRecover import imgRecover
from imageRecover import LASSOBench
prepath = os.path.abspath('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/figs')


if __name__ == '__main__':
    boat = imgRead('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/fishing_boat.bmp')
    pixels = [50, 40, 30, 20, 10]
    testAlphas = np.exp(np.linspace(np.log(10**-6), np.log(10**6), 36))
    minError = []
    blkSize = 8
    B = generateBasisMatrix(blkSize)
    for i in range(len(pixels)):
        S = pixels[i]
        minError.append([])
        p = float(((blkSize ** 2) - S) / (blkSize ** 2))
        for y in range(25):
            minError[i].append([])
            for x in range(24):
                b = imgBlock(boat, blkSize, blkSize * y, blkSize * x)
                c = compressSimulate(b, p)
                A, D = deletePoints(B, c)
                alphaBest, mses = LASSOBench(B, A, D)
                minError[i][y].append(alphaBest)
                print(S, y, x)
                # plt.scatter(testAlphas, mses, c='black', s=8)
                # plt.xlabel("log(alpha)")
                # plt.ylabel("MSE")
                # plt.title("MSE vs. Regularization Parameter, S = " + str(S))
                # plt.axvline(x=alphaBest, color='red', label='Min{MSE} occurs @ alpha = '
                #                                             + str(alphaBest.round(7)))
                # plt.legend()
                #plt.show()
                # plt.savefig(prepath + "natureAlphasS" + str(S) + "x" + str(x) + "y" + str(y))
                # plt.clf()
            # for x in range(1, 3):
            #     imgRecover(boat, 8, S, minError[i][y - 1][x-1], x, y, 'boatcrop')
        np.savetxt('bestAlphasSLASSO' + str(S) + '.csv', minError[i], delimiter=',')

    print(minError)

    nature = imgRead('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/nature.bmp')
    pixels = [150, 100, 50, 30, 10]
    testAlphas = np.exp(np.linspace(np.log(10 ** -6), np.log(10 ** 6), 36))
    minError = []
    blkSize = 16
    B = generateBasisMatrix(blkSize)
    for i in range(len(pixels)):
        S = pixels[i]
        minError.append([])
        p = float(((blkSize ** 2) - S) / (blkSize ** 2))
        for y in range(32):
            minError[i].append([])
            for x in range(40):
                b = imgBlock(nature, blkSize, blkSize * y, blkSize * x)
                c = compressSimulate(b, p)
                A, D = deletePoints(B, c)
                alphaBest, mses = LASSOBench(B, A, D)
                minError[i][y].append(alphaBest)
                print(S, y, x)
                # plt.scatter(testAlphas, mses, c='black', s=8)
                # plt.xlabel("log(alpha)")
                # plt.ylabel("MSE")
                # plt.title("MSE vs. Regularization Parameter, S = " + str(S))
                # plt.axvline(x=alphaBest, color='red', label='Min{MSE} occurs @ alpha = '
                #                                             + str(alphaBest.round(7)))
                # plt.legend()
                # plt.show()
                # plt.savefig(prepath + "natureAlphasS" + str(S) + "x" + str(x) + "y" + str(y))
                # plt.clf()
            # for x in range(1, 3):
            #     imgRecover(boat, 8, S, minError[i][y - 1][x-1], x, y, 'boatcrop')
        np.savetxt('natureBestAlphasSLASSO' + str(S) + '.csv', minError[i], delimiter=',')

    # nature = imgRead('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/nature.bmp')
    # pixels = [150, 100, 50, 30, 10]
    # pixels = [50, 30, 10]
    # alphas = np.linspace(np.log(10 ** -6), np.log(10 ** 6), 36)
    # minError2 = []
    # for i in range(len(pixels)):
    #     S = pixels[i]
    #     minError2.append([])
    #     for y in range(1, 33):
    #         minError2[i].append([])
    #         for x in range(1, 41):
    #             mses = []
    #             for alpha in alphas:
    #                 mse = imgRecover(nature, 16, S, np.exp(alpha), x, y, 'naturecrop', benchmark=True)
    #                 mses.append(mse)
    #             least = alphas[mses.index(min(mses))]
    #             minError2[i][y - 1].append(np.exp(least))
    #             # plt.scatter(alphas, mses, c='black', s=8)
    #             # plt.xlabel("log(alpha)")
    #             # plt.ylabel("MSE")
    #             # plt.title("MSE vs. Regularization Parameter, S = " + str(S))
    #             # plt.axvline(x=least, color='red', label='Min{MSE} occurs @ alpha = '
    #             #                                         + str(np.exp(least).round(7)))
    #             # plt.legend()
    #             # plt.savefig(prepath + "natureAlphasS" + str(S) + "x" + str(x) + "y" + str(y))
    #             # plt.clf()
    #             print("S = " + str(S) + ", y = " + str(y) + ", x = " + str(x))
    #         # for x in range(1, 3):
    #         #     imgRecover(nature, 16, S, minError2[i][y - 1][x - 1], x, y, 'naturecrop')
    #     np.savetxt('natureBestAlphasS' + str(S) + '.csv', minError2[i], delimiter=',')
    #
    # print(minError2)
    # # cropped = imgBlock(boat, 8, 17, 25)  # x = 17, y = 25
    # # pixels = [50, 40, 30, 20, 10]
    # # alphas = np.exp(np.linspace(np.log(10 ** -6), np.log(10 ** 6), 36))
    # # for S in pixels:
    # #     mses = []
    # #     for alpha in alphas:
    # #         mses.append(imgRecover(cropped, 8, S, alpha, 17, 25, '', True))
    # #     least = alphas[mses.index(min(mses))]
    # #     plt.scatter(np.log(alphas), mses, c='black', s=8)
    # #     plt.xlabel("log(alpha)")
    # #     plt.ylabel("MSE")
    # #     plt.title("MSE vs. Regularization Parameter, S = " + str(S))
    # #     plt.axvline(x=np.log(least), color='red', label='Min{MSE} occurs @ alpha = '
    # #                                                     + str(least.round(7)))
    # #     plt.legend()
    # #     plt.savefig(prepath + "boatAlphasS" + str(S))
    # #     plt.clf()
    # #
    # # nature = imgRead('/Users/benmatz/Box/Duke/Spring2023/ECE 580/Projects/MP1/data/nature.bmp')
    # # cropped = imgBlock(nature, 16, 17, 193)  # x = 17, y = 25
    # # pixels = [150, 100, 50, 30, 10]
    # # alphas = np.exp(np.linspace(np.log(10 ** -6), np.log(10 ** 6), 36))
    # # for S in pixels:
    # #     mses = []
    # #     for alpha in alphas:
    # #         mses.append(imgRecover(cropped, 16, S, alpha, 17, 25, '', True))
    # #     least = alphas[mses.index(min(mses))]
    # #     plt.scatter(np.log(alphas), mses, c='black', s=8)
    # #     plt.xlabel("log(alpha)")
    # #     plt.ylabel("MSE")
    # #     plt.title("MSE vs. Regularization Parameter, S = " + str(S))
    # #     plt.axvline(x=np.log(least), color='red', label='Min{MSE} occurs @ alpha = '
    # #                                                     + str(least.round(7)))
    # #     plt.legend()
    # #     plt.savefig(prepath + "natureAlphasS" + str(S))
    # #     plt.clf()