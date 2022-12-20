
import numpy as np
import cv2
from ConvFunctions import conv2

""" Blending images """
def imgBlending(laplacianFGimage, laplacianBGimage, gaussianMask, num_layers):
    lSamples = []
    for LF, LB, mask in zip(laplacianFGimage, laplacianBGimage,gaussianMask):
        LS = LF * mask + LB * (1 - mask)
        lSamples.append(np.float32(LS))

    laplacianBGlap = lSamples[0]

    for i in range(1, num_layers):
        laplacianBGlap = conv2(upSampler(laplacianBGlap, lSamples[i]), w)
        laplacianBGlap = cv2.add(laplacianBGlap, lSamples[i])

    blendedImage = np.clip(laplacianBGlap, 0, 255).astype(np.uint8)

    return blendedImage

""" Gaussian pyramid for mask """
def createGaussianPyrmask(mask,num_layers):
    gaussianMask = np.copy(mask).astype(np.float32)
    gaussianPyrmask = [gaussianMask]
    for i in range(num_layers-1):
        gaussianMask = downSampler(conv2(gaussianMask,w))
        gaussianPyrmask.append(gaussianMask)
    gaussianPyrmask.reverse()
    
    return gaussianPyrmask

""" up and down sampling using nearest neighbor interpolation """
def upSampler(samplingImage,conImage):
    originalRows, originalColumns, c = samplingImage.shape #size of the 1 input image
    newRows, newColumns, c = conImage.shape #size of the 2 input image
    rowRatio = newRows / originalRows
    columnRatio = newColumns / originalColumns
    upscalingRow = (np.floor((np.arange(0,newRows,1)) / rowRatio)).astype(np.int16) #arranges the new number of index that is to be selected
    upscalingColumns = (np.floor((np.arange(0,newColumns,1)) / columnRatio)).astype(np.int16) #arranges the new number of index that is to be selected

    upScaledImgage = samplingImage[upscalingRow , :] #selected the pixel values using the new upscaled index from the original image
    upScaledImgage = upScaledImgage[: , upscalingColumns] #selected the pixel values using the new upscaled index from the original image
    
    return upScaledImgage                                   # return upScaledImgage to pyr_up

def downSampler(samplingImage):
    originalRows, originalColumns, c= samplingImage.shape #size of the input image
    newRows = np.floor(originalRows / 2)
    newColumns = np.floor(originalColumns / 2)
    rowsRatio = newRows / originalRows
    columnsRatio = newColumns / originalColumns
    downscalingRows = np.floor((np.arange(0,newRows,1)) / rowsRatio).astype(np.int16) #arranges the new number of index that is to be selected
    downscalingColumns = np.floor((np.arange(0,newColumns,1)) / columnsRatio).astype(np.int16) #arranges the new number of index that is to be selected

    downScaledImgage = samplingImage[downscalingRows , :] #selected the pixel values using the new downscaled index from the original image
    downScaledImgage = downScaledImgage[: , downscalingColumns] #selected the pixel values using the new downscaled index from the original image

    return downScaledImgage

""" function compute Gaussian/Laplacian pyramid """
def computePyramid(initialImage,num_layers):
    
    global pyramidLayers, w
    pyramidLayers = 1
    imageShape = initialImage.shape[0]
    
    for i in range(num_layers-1):
        if (imageShape//2 < 5):
            print("Layers =  %d " %pyramidLayers)
            num_layers = pyramidLayers
            break
        else:
            imageShape = imageShape // 2
            pyramidLayers += 1

    """ Gaussian kernel """
    gaussianKernel = cv2.getGaussianKernel(5, 2)
    w = gaussianKernel@gaussianKernel.T
    w /= w.sum()

    """ Gaussian pyramid """
    g = initialImage.copy().astype(np.float32)
    gaussianPyramidimg = [g]
    for i in range(num_layers-1):
        g = downSampler(conv2(g,w))
        gaussianPyramidimg.append(g)
        
    """ Laplacian pyramid """
    laplacianPyramidimg = [gaussianPyramidimg[num_layers-1]]
    for i in range(num_layers-1,0,-1):
        G = conv2(upSampler(gaussianPyramidimg[i],gaussianPyramidimg[i-1]),w)
        l = np.subtract(gaussianPyramidimg[i-1], G)
        laplacianPyramidimg.append(l)
    
    return gaussianPyramidimg, laplacianPyramidimg

""" function to align f_img and backGroundimg """
def align(foreGroundimg,backGroundimg,row,col):
    
    alignImage = np.zeros(backGroundimg.shape)
    alignImage[row:(foreGroundimg.shape[0]+row) , col:(foreGroundimg.shape[1]+col)] = foreGroundimg
    return alignImage.astype(np.uint8)