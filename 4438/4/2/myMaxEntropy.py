import SimpleITK as sitk
import numpy as np

def myMaxEntropy(img_bw):
    
    # to calculate the entropy, we need to compute the normalized histogram first
    bw_histogram = np.histogram( img_bw, bins=256, range=[0,256] )
    PMF = 1.0/np.sum(bw_histogram[0]) * bw_histogram[0]

    # Compute the cumulative density function CDF
    bw_histogram_normalized=PMF
    CDF = np.zeros(len(bw_histogram_normalized))
    CDF[0] = bw_histogram_normalized[0]
    for i in range(1,len(bw_histogram_normalized)):
        CDF[i] = CDF[i-1] + bw_histogram_normalized[i]
    
    hl = np.zeros(256)  #entropy for (black) class below the threshold value
    hh = np.zeros(256)  #entropy for (white) class above the threshold value
    h = np.zeros(256)   #total entropy, sum of the two classes

    # For a given pixel intensity  i,i∈[0,t], compute Hl(i) and Hh(i) and define H=Hl+Hh   
    for t in range(0,len(hl)):
        cl = CDF[t]
        if ( cl > 0 ):
            for i in range(0,t+1):
                if (bw_histogram_normalized[i] > 0):
                    hl[t] = hl[t] - (bw_histogram_normalized[i]/cl) * np.log2(bw_histogram_normalized[i]/cl)
            
        ch = 1.0-cl
        if ( ch > 0 ):
            for i in range(t+1,256):
                if ( bw_histogram_normalized[i] > 0 ):
                    hh[t] = hh[t] - ( bw_histogram_normalized[i]/ch) * np.log2(bw_histogram_normalized[i]/ch)
                
        h[t] = hl[t] + hh[t] # total entropy
    EntropyLevel = int(np.argmax(h))  # find the maximum entropy

    print('Maximum entropy occurs at intensity:', EntropyLevel)
        
    MaxEntropySeg = img_bw > EntropyLevel
    # any pixel in the input grayscale image with intensity higher than or equal to the computed threshold value is converted to 1 (foreground),lower than the computed threshold value is converted to 0 (background)
    
    MaxEntropySeg.CopyInformation(img_bw)    # copy information like pixel size, spacing, direction etc.
    
    MaxEntropySeg = sitk.Cast(sitk.RescaleIntensity(MaxEntropySeg), sitk.sitkUInt8)    #cast to UInt8
    return MaxEntropySeg