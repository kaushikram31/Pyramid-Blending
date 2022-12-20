import numpy as np
import cv2
from my_functions import align, computePyramid, createGaussianPyrmask, imgBlending

""" function to create mask """

drawing = False
ix, iy = -1, -1
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, drawing, mode, ax, ay, zx, zy
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        ax, ay = x, y
        print("starting", ix, iy)
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.rectangle(f_imgcp, (ix, iy), (x, y), (203, 220, 203), -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        zx, zy = x, y
        print("ending", x, y)

def draw_ellipse(event, x, y, flags, param):  # openCV defaut parameters
    global ix, iy, drawing, ax, ay, zx, zy  # these vars can be used outside of this function
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        ax, ay = x, y
        print("starting", ix, iy)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            cv2.ellipse(f_imgcp, ((ix + (x - ix) // 2), (iy + (y - iy) // 2)), (x - ix, y - iy),0, 0, 360, (0,0,0), -1)

    elif event == cv2.EVENT_LBUTTONUP:  # left mouse button is released
        drawing = False
        zx, zy = x, y
        print("ending", x, y)

foreGroundimg = cv2.resize(cv2.imread(r'D:\NCSU\NCSU Courses\DIS - 558\Image-Blending-main\ak_images\pair 3\m1.jpg'),(500,450))
og = cv2.resize(cv2.imread (r'D:\NCSU\NCSU Courses\DIS - 558\Image-Blending-main\ak_images\pair 3\m1.jpg'),(500,450))
backGroundimg = cv2.resize(cv2.imread(r'D:\NCSU\NCSU Courses\DIS - 558\Image-Blending-main\ak_images\pair 3\m2.jpg'),(500,450))

if (backGroundimg.shape[0] > foreGroundimg.shape[0]):
    foreGroundimg_align = align(foreGroundimg, backGroundimg, 50, 35)
    foreGroundimg = np.copy(foreGroundimg_align)
    f_imgcp = np.copy(foreGroundimg_align)
else:
    f_imgcp = np.copy(foreGroundimg)



"""   Taking inputs from the user   """

shape = int(input("Shape of the mask you want to draw:\nEnter 0 if rectangle\nEnter 1 if ellipse\n"))
cv2.namedWindow('image')
if shape == 0:
    rec_draw = True
    cv2.setMouseCallback('image', draw_rectangle)
elif shape == 1:
    rec_draw = False
    cv2.setMouseCallback('image', draw_ellipse)


while (1):
    cv2.imshow('image', f_imgcp)
    if cv2.waitKey(20) & 0xFF == 27:
        break


def create_mask_fimg(f_imgcp, init_x, init_y, new_x, new_y):
    new_mask = np.zeros(f_imgcp.shape).astype(np.float32)

    """ OpenCV function to draw rectangle and ellipse """
    if rec_draw == False:
        new_mask = cv2.ellipse(new_mask, ((init_x + (new_x - init_x) // 2), (init_y + (new_y - init_y) // 2)),
                               (new_x - init_x, new_y - init_y), 0, 0, 360, (1, 1, 1), -1)
    else:
        new_mask = cv2.rectangle(new_mask, (init_x, init_y), (new_x, new_y), (1, 1, 1), -1)

    return new_mask

mask = create_mask_fimg(f_imgcp, ax, ay, zx, zy)


cv2.namedWindow('Foreground Image', cv2.WINDOW_NORMAL)
cv2.imshow('Foreground Image', foreGroundimg)
cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
cv2.imshow('Mask', mask)
cv2.namedWindow('Background Image', cv2.WINDOW_NORMAL)
cv2.imshow('Background Image', backGroundimg)
cv2.waitKey(0)
cv2.destroyAllWindows()

""" compute Gaussian/Laplacian pyramid, create Gaussian mask, and blend images """
gPyrfimg, lPyrfimg = computePyramid(foreGroundimg, 10)
gPyrbimg, lPyrbimg = computePyramid(backGroundimg, 10)
gPyrmask = createGaussianPyrmask(mask, len(gPyrfimg))
blended_img = imgBlending(lPyrfimg, lPyrbimg, gPyrmask, len(gPyrfimg))

# cv2.namedWindow('Foreground Image', cv2.WINDOW_NORMAL)
# cv2.imshow('Foreground Image', foreGroundimg)
# cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
# cv2.imshow('Mask', mask)
# cv2.namedWindow('Background Image', cv2.WINDOW_NORMAL)
# cv2.imshow('Background Image', backGroundimg)
cv2.namedWindow('Blended Image', cv2.WINDOW_NORMAL)
cv2.imshow('Blended Image', blended_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

""" write result images for report """
#cv2.imwrite(r'D:\PyCharm Community Edition 2022.2.2\ImageBlending\foreGroundimg.png', f_imgcp)
#cv2.imwrite(r'D:\PyCharm Community Edition 2022.2.2\ImageBlending\backGroundimg.png', backGroundimg)
cv2.imwrite(r'D:\NCSU\NCSU Courses\DIS - 558\Image-Blending-main\ak_images\pair 3\blendedImg.png', blended_img)