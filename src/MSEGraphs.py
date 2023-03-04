import numpy as np
import matplotlib.pyplot as plt

boatK = 8
boatS = np.asarray([50, 40, 30, 20, 10])
boatMSE = [82.795, 243.966, 430.215, 738.271, 1103.212]
boatFiltMSE = [153.729, 234.401, 341.245, 539.536, 805.95]
plt.plot(boatS, boatMSE, 'r--', label='Unfiltered')
plt.plot(boatS, boatFiltMSE, 'b', label='Filtered')
plt.xlabel('Sensed Pixels (S)')
plt.ylabel('MSE')
plt.title('Fishing Boat MSE with and without Median Filtering')
plt.xticks(boatS)
plt.legend()
# plt.show()
# plt.savefig('boatMSE')
plt.clf()


natureK = 16
natureS = np.asarray([150, 100, 50, 30, 10])
natureMSE = [215.923, 347.963, 517.379, 616.76, 934.645]
natureFiltMSE = [272.071, 340.04, 445.909, 505.805, 775.932]
plt.plot(natureS, natureMSE, 'g--', label='Unfiltered')
plt.plot(natureS, natureFiltMSE, 'k', label='Filtered')
plt.xlabel('Sensed Pixels (S)')
plt.ylabel('MSE')
plt.title('Nature MSE with and without Median Filtering')
plt.xticks(natureS)
plt.legend()
# plt.show()
# plt.savefig('natureMSE')
plt.clf()


params = {'mathtext.default': 'regular'}
plt.rcParams.update(params)
plt.plot(boatS/(boatK**2), boatMSE, 'r--', label='Boat, Unfiltered')
plt.plot(boatS/(boatK**2), boatFiltMSE, 'b', label='Boat, Filtered')
plt.plot(natureS/(natureK**2), natureMSE, 'g--', label='Nature, Unfiltered')
plt.plot(natureS/(natureK**2), natureFiltMSE, 'k', label='Nature, Filtered')
plt.xlabel('Sensed Pixel Ratio ($S/K^2$)')
plt.ylabel('MSE')
plt.title('MSE vs. $S/K^2$')
plt.legend()
# plt.show()
# plt.savefig('bothMSE')
plt.clf()