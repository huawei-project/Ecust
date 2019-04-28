import cv2
import random
import numpy as np
from matplotlib import pyplot as plt

#高斯噪声
def gaussianNoise(src, means, sigma, percetage):
    dst = src.copy()
    num = int(percetage*src.shape[0]*src.shape[1])
    for i in range(num):
        randX = random.randint(0, src.shape[0]-1)
        randY = random.randint(0, src.shape[1]-1)
        dst[randX, randY] = dst[randX, randY] + random.gauss(means, sigma)
        dst[dst<0] = 0
        dst[dst>255] = 255
    return dst

def signal_to_noise_ratio(oriImg, noisedImg):
    
    signal = np.sum(oriImg**2)
    noise  = np.sum((oriImg - noisedImg)**2)
    ratio  = 10 * np.log10(signal / noise)

    return ratio

def drawGaussian(mean, sigma):
    """ 显示一维高斯图像
    """
    gaussian = lambda x, mean, sigma: np.exp(- (x - mean)**2 / (2 * sigma**2)) / np.sqrt(2*np.pi) / sigma
    x = np.linspace(-255, 255, 2*255)
    y = gaussian(x, mean, sigma)
    plt.figure("mu: {:.2f}, sigma: {:.2f}".format(mean, sigma))
    plt.plot(x, y)
    plt.show()

if __name__ == "__main__":
    drawGaussian(0, 75)

    img = cv2.imread('test.png')
    img = cv2.resize(img, (160, 120)[::-1])
    img = cv2.cvtColor(img ,cv2.COLOR_BGR2GRAY)
    img_01 = gaussianNoise(img, 0, 75, 0.1)
    img_02 = gaussianNoise(img, 0, 75, 0.2)
    img_03 = gaussianNoise(img, 0, 75, 0.3)
    img_04 = gaussianNoise(img, 0, 75, 0.4)
    img_05 = gaussianNoise(img, 0, 75, 0.5)
    img_06 = gaussianNoise(img, 0, 75, 0.6)
    img_07 = gaussianNoise(img, 0, 75, 0.7)
    img_08 = gaussianNoise(img, 0, 75, 0.8)
    img_09 = gaussianNoise(img, 0, 75, 0.9)
    img_10 = gaussianNoise(img, 0, 75, 1.0)

    fig = plt.figure("images")
    sp1 = fig.add_subplot(431)
    sp1.imshow(img_01)
    sp2 = fig.add_subplot(432)
    sp2.imshow(img_02)
    sp3 = fig.add_subplot(433)
    sp3.imshow(img_03)
    sp4 = fig.add_subplot(434)
    sp4.imshow(img_04)
    sp5 = fig.add_subplot(435)
    sp5.imshow(img_05)
    sp6 = fig.add_subplot(436)
    sp6.imshow(img_06)
    plt.show()

    snr_01 = signal_to_noise_ratio(img, img_01)
    snr_02 = signal_to_noise_ratio(img, img_02)
    snr_03 = signal_to_noise_ratio(img, img_03)
    snr_04 = signal_to_noise_ratio(img, img_04)
    snr_05 = signal_to_noise_ratio(img, img_05)
    snr_06 = signal_to_noise_ratio(img, img_06)
    snr_07 = signal_to_noise_ratio(img, img_07)
    snr_08 = signal_to_noise_ratio(img, img_08)
    snr_09 = signal_to_noise_ratio(img, img_09)
    snr_10 = signal_to_noise_ratio(img, img_10)

    y = [snr_01, snr_02, snr_03, snr_04, snr_05, snr_06, snr_07, snr_08, snr_09, snr_10]
    x = [(i+1)*0.1 for i in range(10)]
    plt.figure("snr-percentage")
    plt.plot(x, y)
    plt.show()