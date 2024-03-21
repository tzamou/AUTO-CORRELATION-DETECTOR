import numpy as np
from scipy import signal
import cv2
import matplotlib.pyplot as plt


class AutoCorrelationDetector:
    def __init__(self,
                 imgpath: str,
                 window_size: int=3):
        img = cv2.imread(imgpath)
        self.img = cv2.cvtColor(img, code=cv2.COLOR_BGR2GRAY)
        self.s = np.array([[-1, 0, 1]])
        self.st = self.s.T
        self.w = self.init_w(window_size)

    def init_w(self, window_size):
        w = np.zeros(shape=(window_size, window_size))
        for u in range(w.shape[0]):
            for v in range(w.shape[1]):
                w[u, v] = np.exp(-(u**2+v**2))/(2*3**2) # sigma=3
        return w
    def get_eigenvalues(self, A, B, C):
        alpha = np.zeros(shape=self.img.shape)
        beta = np.zeros(shape=self.img.shape)

        for i in range(alpha.shape[0]):
            for j in range(alpha.shape[1]):
                M = np.array([[A[i, j], C[i, j]], [C[i, j], B[i, j]]])
                alpha[i, j], beta[i, j] = np.linalg.eigvals(M)

        return alpha, beta
    def get_corner_resopnse(self, k):
        self.k = k
        X = signal.convolve2d(self.img, self.s, mode='same')
        Y = signal.convolve2d(self.img, self.st, mode='same')
        A = signal.convolve2d(X*X, self.w, mode='same')
        B = signal.convolve2d(Y*Y, self.w, mode='same')
        C = signal.convolve2d(X*Y, self.w, mode='same')
        # alpha, beta = self.get_eigenvalues(A, B, C)
        # TR = alpha+beta
        # Det_M = alpha*beta
        # R = (Det_M-k*TR**2)
        R = A*B-C**2-k*(A+B)**2
        return R
    def plot_edge(self, R, th_start=1, th_end=100, img_num=100):
        for i, th in enumerate(np.linspace(th_start, th_end, img_num), start=1):
            print(f'\rProcessing edge image {i}/{img_num}.', end='')
            th *= -1
            edge = R < th
            plt.title("Image's Edge", fontsize=20)
            plt.xlabel(f'k = {self.k} th = {th:.2f}', fontsize=12)
            plt.imshow(edge, cmap='gray')
            plt.savefig(f'./Result/AutoCorrelationDetector/edge/Result{i:03d}.png')
            plt.clf()
            plt.close()
        print()

    def plot_corner(self, R, th_start=1, th_end=100, img_num=100):
        for i, th in enumerate(np.linspace(th_start, th_end, img_num), start=1):
            print(f'\rProcessing corner image {i}/{img_num}.', end='')
            th /= 100
            corner = R > th
            plt.title("Image's Corner", fontsize=20)
            plt.xlabel(f'k = {self.k} th = {th:.2f}', fontsize=12)
            plt.imshow(corner, cmap='gray')
            plt.savefig(f'./Result/AutoCorrelationDetector/corner/Result{i:03d}.png')
            plt.clf()
            plt.close()
        print()


    def plot_result(self, R, th_e, th_c, mode='show', save_path=None):
        edge = R < th_e
        corner = R > th_c
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].set_title("Original Image", fontsize=20)
        ax[0].imshow(self.img, cmap='gray')

        ax[1].set_title("Image's Corner", fontsize=20)
        ax[1].set_xlabel(f'k = {self.k}\nth = {th_c}', fontsize=12)
        ax[1].imshow(corner, cmap='gray')

        ax[2].set_title("Image's Edge", fontsize=20)
        ax[2].set_xlabel(f'k = {self.k}\nth = {th_e}', fontsize=12)
        ax[2].imshow(edge, cmap='gray')
        if mode == 'show':
            plt.show()
        elif mode == 'save':
            assert save_path!=None, 'Please input the save path.'
            plt.savefig(save_path)
            plt.clf()
            plt.close()
        else:
            raise ValueError("The argument 'mode' must be 'show' or 'save'.")

if __name__ == '__main__':
    acd = AutoCorrelationDetector(imgpath='./src/test2.jpg')
    R = acd.get_corner_resopnse(k=0.04)
    acd.plot_result(R=R, th_c=np.max(R)/100, th_e=-np.max(R)/1000,
                    mode='save',
                    save_path='./result/AutoCorrelationDetector/result.png')


