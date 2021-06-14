# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 16:54:14 2021

@author: ShohanurSobuj
"""


'''!pip install bm3d'''
#!pip install bm3d
import bm3d
import cv2
from skimage import io, img_as_float
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio


img = img_as_float(io.imread("oct.jpeg", as_gray=True))
BM3D_denoised_image = bm3d.bm3d(img, sigma_psd=0.2, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
#save the denoised image
io.imsave("denoised.jpg",BM3D_denoised_image)

"""
bm3d library is not well documented yet, but looking into source code....
sigma_psd - noise standard deviation
stage_arg: Determines whether to perform hard-thresholding or Wiener filtering.
stage_arg = BM3DStages.HARD_THRESHOLDING or BM3DStages.ALL_STAGES (slow but powerful)
All stages performs both hard thresholding and Wiener filtering. 
"""

cv2.imshow("Original", img)
cv2.imshow("Denoised", BM3D_denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#check histogram
plt.hist(img.flat,bins=100,range=(0,1))
plt.hist(BM3D_denoised_image.flat,bins=100,range=(0,1))


mse_error = mean_squared_error(img, BM3D_denoised_image)
print(mse_error)

psnr = peak_signal_noise_ratio(img, BM3D_denoised_image)
print(psnr)