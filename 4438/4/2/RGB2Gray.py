import SimpleITK as sitk
import numpy as np

def RGB2Gray(img):
    img = sitk.ReadImage(img)    # Read image
    sitk.Show(img)
    img_Vector = sitk.GetArrayFromImage(img)    # convert the input image to a numpy array
    img_bw = sitk.GetImageFromArray(.299 * img_Vector[:,:,0] + .587 * img_Vector[:,:,1] + .114 * img_Vector[:,:,2])
    # calculate weighted average of the R-G-B channels and generate a 2D image gray scale. 3D numpy array becuase of RGB three channels
    img_bw.CopyInformation(img)    # copy information like pixel size, spacing, direction etc.
    
    img_bw = sitk.Cast(sitk.RescaleIntensity(img_bw), sitk.sitkUInt8)    #cast to UInt8
    img_bw = sitk.Cast(img_bw, sitk.sitkUInt8)    #cast to UInt8
    return img_bw
