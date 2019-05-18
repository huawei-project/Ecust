import cv2
import random
import numpy as np
import skimage
from matplotlib import pyplot as plt

# 高斯噪声


def gaussianNoise(src, means, sigma, percetage):
    dst = src.copy()
    num = int(percetage * src.shape[0] * src.shape[1])
    for i in range(num):
        randX = random.randint(0, src.shape[0] - 1)
        randY = random.randint(0, src.shape[1] - 1)
        dst[randX, randY] = dst[randX, randY] + random.gauss(means, sigma)
        dst[dst < 0] = 0
        dst[dst > 255] = 255
    return dst

# 椒盐噪声


def spNoise(image, amount):
    """
    Parameters:
        image:  {ndarray(H, W, C)}
        amount: {float} roportion of image pixels to replace with noise on range [0, 1]
    Notes:
        Function to add random noise of various types to a floating-point image.
    """
    dtype = image.dtype
    if len(image.shape) == 3:
        noisedImage = np.zeros_like(image)
        for i in range(3):
            noisedImage[:, :, i] = skimage.util.random_noise(
                image[:, :, i], mode='s&p', amount=amount)
    elif len(image.shape) == 2:
        noisedImage = skimage.util.random_noise(
            image, mode='s&p', amount=amount)
    else:
        raise TypeError
    noisedImage = (noisedImage * 255).astype(dtype)
    return noisedImage


def addsalt_pepper(img, SNR):
    if len(img.shape) == 2:
        img = np.expand_dims(img, -1)
    img_ = img.copy()
    h, w, c = img_.shape
    mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[
                            SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    mask = np.repeat(mask, c, axis=-1)  # 按channel 复制到 与img具有相同的shape
    img_[mask == 1] = 255  # 盐噪声
    img_[mask == 2] = 0  # 椒噪声
    return np.squeeze(img_)


def signal_to_noise_ratio(oriImg, noisedImg):

    signal = np.sum(oriImg**2)
    noise = np.sum((oriImg - noisedImg)**2)
    ratio = 10 * np.log10(signal / noise)

    return ratio


def drawGaussian(mean, sigma):
    """ 显示一维高斯图像
    """
    def gaussian(x, mean, sigma): return np.exp(- (x - mean) **
                                                2 / (2 * sigma**2)) / np.sqrt(2 * np.pi) / sigma
    x = np.linspace(-255, 255, 2 * 255)
    y = gaussian(x, mean, sigma)
    plt.figure("mu: {:.2f}, sigma: {:.2f}".format(mean, sigma))
    plt.plot(x, y)
    plt.show()


if __name__ == "__main__":

    img = cv2.imread('W750E500_750.JPG')
    img = cv2.resize(img, (120, 160)[::-1])
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_01 = gaussianNoise(img, 0, 75, 0.1)
    # img_02 = gaussianNoise(img, 0, 75, 0.2)
    # img_03 = gaussianNoise(img, 0, 75, 0.3)
    # img_04 = gaussianNoise(img, 0, 75, 0.4)
    # img_05 = gaussianNoise(img, 0, 75, 0.5)
    # img_06 = gaussianNoise(img, 0, 75, 0.6)
    # img_07 = gaussianNoise(img, 0, 75, 0.7)
    # img_08 = gaussianNoise(img, 0, 75, 0.8)
    # img_09 = gaussianNoise(img, 0, 75, 0.9)
    # img_10 = gaussianNoise(img, 0, 75, 1.0)

    noise = [0.006, 0.008, 0.010, 0.012,
             0.014, 0.016, 0.018, 0.020, 0.04, 0.06]

    NoiseFunc = addsalt_pepper
    img_01 = NoiseFunc(img, 1 - 0.006)
    img_02 = NoiseFunc(img, 1 - 0.008)
    img_03 = NoiseFunc(img, 1 - 0.010)
    img_04 = NoiseFunc(img, 1 - 0.012)
    img_05 = NoiseFunc(img, 1 - 0.014)
    img_06 = NoiseFunc(img, 1 - 0.016)
    img_07 = NoiseFunc(img, 1 - 0.018)
    img_08 = NoiseFunc(img, 1 - 0.020)
    img_09 = NoiseFunc(img, 1 - 0.040)
    img_10 = NoiseFunc(img, 1 - 0.060)

    fig = plt.figure(figsize=(10, 10))
    sp1 = fig.add_subplot(331)
    sp1.imshow(img_01)
    sp2 = fig.add_subplot(332)
    sp2.imshow(img_02)
    sp3 = fig.add_subplot(333)
    sp3.imshow(img_03)
    sp4 = fig.add_subplot(334)
    sp4.imshow(img_04)
    sp5 = fig.add_subplot(335)
    sp5.imshow(img_05)
    sp6 = fig.add_subplot(336)
    sp6.imshow(img_06)
    sp7 = fig.add_subplot(337)
    sp7.imshow(img_07)
    sp8 = fig.add_subplot(338)
    sp8.imshow(img_08)
    sp9 = fig.add_subplot(339)
    sp9.imshow(img_09)
    plt.savefig('noised_image.png')
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

    y = [snr_01, snr_02, snr_03, snr_04, snr_05,
         snr_06, snr_07, snr_08, snr_09, snr_10]
    #x = [(i + 1) * 0.1 for i in range(10)]
    plt.figure("nsr-percentage")
    plt.plot(noise, y)
    plt.show()
