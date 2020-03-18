import SimpleITK as sitk
import numpy as np

def myOtsuThresholding(img_bw):
    n = np.histogram( img_bw, bins=256, range=[0,256] )    # Generate histogram from converted grayscale image 
    p = n[0]/np.sum(n[0])    # bw_histogram_normalized = 1.0/(bw_histogram[0].sum()) * bw_histogram[0]
    CDF = np.zeros( len( p ) )    # Compute the cumulative distribution function
    CDF[0] = p[0]
    for i in range(1, len(p )):
        CDF[i] = CDF[i-1] + p[i]

    w_0 = CDF
    w_1 = 1-w_0

    levels = np.zeros([256])

    w = np.zeros([256])     # 0th-order cumulative moment, or CDF, we computed it earlier but repeating it for completeness
    mu = np.zeros([256])    # 1st-order cumulative moment, or the class mean
    w[0] = p[0]
    mu[0] = 0

    # preparation
    for t in range(1,256):    # for each threshold value t, compute the 0th and 1st moment
        w[t] = w[t-1] + p[t]
        mu[t] = mu[t-1] + t * p[t]
    mu_T = mu[255]            # the total mean pixel intensity of the original image
    # exhustive search
    for t in range(0,256):
        if ( w[t]*(1-w[t]) > 0 ):   # avoid division by 0
            levels[t] = ((mu_T * w[t] - mu[t])*(mu_T * w[t] - mu[t])) /(w[t]*(1-w[t]))
    level = int(np.argmax(levels))    # find the maximum level
    
    print(level)
    

    OtsuSeg = img_bw > level
    # any pixel in the input grayscale image with intensity higher than or equal to the computed threshold value is converted to 1 (foreground),lower than the computed threshold value is converted to 0 (background)
    
    # Numpy's method to calculate variance is biased (divided by n) which is used to calculate variance of a know sequence. Since we are working with images, and we assumes that the pixel intensity follows a stochastic process, dividing instead by  (n−1) yields an unbiased estimator.
    
    OtsuSeg.CopyInformation(img_bw)    # copy information like pixel size, spacing, direction etc.
    
    OtsuSeg = sitk.Cast(sitk.RescaleIntensity(OtsuSeg), sitk.sitkUInt8)    #cast to UInt8
    return OtsuSeg