import numpy as np
import cv2

""" padding function """


def padSize(box):
    box_size = box.shape[0]
    if (box_size % 2 == 0):
        ans = box_size // 2;
    else:
        ans = (box_size - 1) // 2

    # print('padsize', ans)
    return ans


def wraparound(padsize, B, G, R, img_grey, inp):
    if inp == 1:
        n = padsize
        sizex = len(B)
        sizey = len(B[0])
        new_img = np.zeros((sizex + 2 * n, sizey + 2 * n, 3))
        B_1 = new_img[:, :, 0]
        G_1 = new_img[:, :, 1]
        R_1 = new_img[:, :, 2]
        x = B.shape[0]
        y = B.shape[1]
        xi = B_1.shape[0]
        yi = B_1.shape[1]

        # botton placed on top
        w = B[x - n:x, :]
        B_1[0:n, n:yi - n] = w[::-1]
        w = G[x - n:x, :]
        G_1[0:n, n:yi - n] = w[::-1]
        w = R[x - n:x, :]
        R_1[0:n, n:yi - n] = w[::-1]

        # top placed in botton
        B_1[xi - n:xi, n:yi - n] = B[0:n, :]
        G_1[xi - n:xi, n:yi - n] = G[0:n, :]
        R_1[xi - n:xi, n:yi - n] = R[0:n, :]

        # right placed on left
        B_1[n:xi - n, 0:n] = B[:, y - n:y]
        G_1[n:xi - n, 0:n] = G[:, y - n:y]
        R_1[n:xi - n, 0:n] = R[:, y - n:y]

        # left placed on right
        B_1[n:xi - n, yi - n:yi] = B[:, 0:n]
        G_1[n:xi - n, yi - n:yi] = G[:, 0:n]
        R_1[n:xi - n, yi - n:yi] = R[:, 0:n]

        # bottom right on top left
        B_1[0:n, 0:n] = B[x - n:x, y - n:y]
        G_1[0:n, 0:n] = G[x - n:x, y - n:y]
        R_1[0:n, 0:n] = R[x - n:x, y - n:y]

        # top left on botton right
        B_1[xi - n:xi, yi - n:yi] = B[0:n, 0:n]
        G_1[xi - n:xi, yi - n:yi] = G[0:n, 0:n]
        R_1[xi - n:xi, yi - n:yi] = R[0:n, 0:n]

        # bottom left on top right
        B_1[0:n, yi - n:yi] = B[x - n:x, 0:n]
        G_1[0:n, yi - n:yi] = G[x - n:x, 0:n]
        R_1[0:n, yi - n:yi] = R[x - n:x, 0:n]

        # top right on botton left
        B_1[xi - n:xi, 0:n] = B[0:n, y - n:y]
        G_1[xi - n:xi, 0:n] = G[0:n, y - n:y]
        R_1[xi - n:xi, 0:n] = R[0:n, y - n:y]

        rx = len(B_1)
        ry = len(B_1[0])
        B_1[n:rx - n, n:ry - n] = B
        G_1[n:rx - n, n:ry - n] = G
        R_1[n:rx - n, n:ry - n] = R
        new_img_cv = np.dstack((B_1, G_1, R_1))
        return new_img_cv

    elif inp == 2:
        n = padsize
        sizex = len(img_grey)
        sizey = len(img_grey[0])
        new_img = np.zeros((sizex + 2 * n, sizey + 2 * n))
        x = img_grey.shape[0]
        y = img_grey.shape[1]
        xi = new_img.shape[0]
        yi = new_img.shape[1]

        # botton placed on top
        w = img_grey[x - n:x, :]
        new_img[0:n, n:yi - n] = w[::-1]

        # top placed in botton
        new_img[xi - n:xi, n:yi - n] = img_grey[0:n, :]

        # right placed on left
        new_img[n:xi - n, 0:n] = img_grey[:, y - n:y]

        # left placed on right
        new_img[n:xi - n, yi - n:yi] = img_grey[:, 0:n]

        # bottom right on top left
        new_img[0:n, 0:n] = img_grey[x - n:x, y - n:y]

        # top left on botton right
        new_img[xi - n:xi, yi - n:yi] = img_grey[0:n, 0:n]

        # bottom left on top right
        new_img[0:n, yi - n:yi] = img_grey[x - n:x, 0:n]

        # top right on botton left
        new_img[xi - n:xi, 0:n] = img_grey[0:n, y - n:y]

        rx = len(new_img)
        ry = len(new_img[0])
        new_img[n:rx - n, n:ry - n] = img_grey

        return new_img


def copyedge(padsize, B, G, R, img_grey, inp):
    if (inp == 1):
        n = padsize
        sizex = len(B)
        sizey = len(B[0])
        new_img = np.zeros((sizex + 2 * n, sizey + 2 * n, 3))
        B_1 = new_img[:, :, 0]
        G_1 = new_img[:, :, 1]
        R_1 = new_img[:, :, 2]
        x = len(B) - 1
        y = len(B[0]) - 1
        xi = len(B_1)
        yi = len(B_1[0])

        for i in range(len(B_1)):
            for j in range(len(B_1[0])):
                if (i < n and j < n):
                    B_1[i][j] = B[0][0]
                    G_1[i][j] = G[0][0]
                    R_1[i][j] = R[0][0]

                if (i < n and j >= yi - n):
                    B_1[i][j] = B[0][y]
                    G_1[i][j] = G[0][y]
                    R_1[i][j] = R[0][y]

                if (i >= xi - n and j < n):
                    B_1[i][j] = B[x][0]
                    G_1[i][j] = G[x][0]
                    R_1[i][j] = R[x][0]

                if (i >= xi - n and j >= yi - n):
                    B_1[i][j] = B[x][y]
                    G_1[i][j] = G[x][y]
                    R_1[i][j] = R[x][y]

                if (i < n and j >= n and j < yi - n):
                    B_1[i][j] = B[0][j - n]
                    G_1[i][j] = G[0][j - n]
                    R_1[i][j] = R[0][j - n]

                if (j < n and i >= n and i < xi - n):
                    B_1[i][j] = B[i - n][0]
                    G_1[i][j] = G[i - n][0]
                    R_1[i][j] = R[i - n][0]

                if (i >= n and i < xi - n and j >= yi - n):
                    B_1[i][j] = B[i - n][y]
                    G_1[i][j] = G[i - n][y]
                    R_1[i][j] = R[i - n][y]

                if (j >= n and j < yi - n and i >= xi - n):
                    B_1[i][j] = B[x][j - n]
                    G_1[i][j] = G[x][j - n]
                    R_1[i][j] = R[x][j - n]

        rx = len(B_1)
        ry = len(B_1[0])
        B_1[n:rx - n, n:ry - n] = B
        G_1[n:rx - n, n:ry - n] = G
        R_1[n:rx - n, n:ry - n] = R
        new_img_cv = np.dstack((B_1, G_1, R_1))
        return new_img_cv

    elif inp == 2:

        n = padsize
        sizex = len(img_grey)
        sizey = len(img_grey[0])
        new_img = np.zeros((sizex + 2 * n, sizey + 2 * n))
        x = len(img_grey) - 1
        y = len(img_grey[0]) - 1
        xi = len(new_img)
        yi = len(new_img[0])

        for i in range(len(new_img)):
            for j in range(len(new_img[0])):
                if (i < n and j < n):
                    new_img[i][j] = B[0][0]

                if (i < n and j >= yi - n):
                    new_img[i][j] = B[0][y]

                if (i >= xi - n and j < n):
                    new_img[i][j] = B[x][0]

                if (i >= xi - n and j >= yi - n):
                    new_img[i][j] = B[x][y]

                if (i < n and j >= n and j < yi - n):
                    new_img[i][j] = B[0][j - n]

                if (j < n and i >= n and i < xi - n):
                    new_img[i][j] = B[i - n][0]

                if (i >= n and i < xi - n and j >= yi - n):
                    new_img[i][j] = B[i - n][y]

                if (j >= n and j < yi - n and i >= xi - n):
                    new_img[i][j] = B[x][j - n]

        rx = len(new_img)
        ry = len(new_img[0])
        new_img[n:rx - n, n:ry - n] = img_grey

        return new_img


def reflect(padsize, B, G, R, img_grey, inp):
    if inp == 1:
        n = padsize
        sizex = len(B)
        sizey = len(B[0])
        new_img = np.zeros((sizex + 2 * n, sizey + 2 * n, 3))
        B_1 = new_img[:, :, 0]
        G_1 = new_img[:, :, 1]
        R_1 = new_img[:, :, 2]
        x = B.shape[0]
        y = B.shape[1]
        xi = B_1.shape[0]
        yi = B_1.shape[1]

        # Bottom
        w1b = B[x - n - 1:x - 1, :]
        w1b = w1b[::-1]
        w1g = G[x - n - 1:x - 1, :]
        w1g = w1g[::-1]
        w1r = R[x - n - 1:x - 1, :]
        w1r = w1r[::-1]

        # Top
        w2b = B[1:n + 1, :]
        w2b = w2b[::-1]
        w2g = G[1:n + 1, :]
        w2g = w2g[::-1]
        w2r = R[1:n + 1, :]
        w2r = w2r[::-1]

        # Right
        w3b = B[:, y - n - 1:y - 1]
        w3b = np.fliplr(w3b)
        w3g = G[:, y - n - 1:y - 1]
        w3g = np.fliplr(w3g)
        w3r = R[:, y - n - 1:y - 1]
        w3r = np.fliplr(w3r)

        # Left
        w4b = B[:, 1:n + 1]
        w4b = np.fliplr(w4b)
        w4g = G[:, 1:n + 1]
        w4g = np.fliplr(w4g)
        w4r = R[:, 1:n + 1]
        w4r = np.fliplr(w4r)

        # Top left
        w2lb = w2b[:, 1:n + 1]
        w2lb = np.fliplr(w2lb)
        w2lg = w2g[:, 1:n + 1]
        w2lg = np.fliplr(w2lg)
        w2lr = w2r[:, 1:n + 1]
        w2lr = np.fliplr(w2lr)

        # Top Right
        w2rb = w2b[:, y - n - 1:y - 1]
        w2rb = np.fliplr(w2rb)
        w2rg = w2g[:, y - n - 1:y - 1]
        w2rg = np.fliplr(w2rg)
        w2rr = w2r[:, y - n - 1:y - 1]
        w2rr = np.fliplr(w2rr)

        # Bottom left
        w1lb = w1b[:, 1:n + 1]
        w1lb = np.fliplr(w1lb)
        w1lg = w1g[:, 1:n + 1]
        w1lg = np.fliplr(w1lg)
        w1lr = w1r[:, 1:n + 1]
        w1lr = np.fliplr(w1lr)

        # Bottom right
        w1rb = w1b[:, y - n - 1:y - 1]
        w1rb = np.fliplr(w1rb)
        w1rg = w1g[:, y - n - 1:y - 1]
        w1rg = np.fliplr(w1rg)
        w1rr = w1r[:, y - n - 1:y - 1]
        w1rr = np.fliplr(w1rr)

        # Placing on new matrix
        # top
        B_1[0:n, n:yi - n] = w2b
        G_1[0:n, n:yi - n] = w2g
        R_1[0:n, n:yi - n] = w2r

        # botton
        B_1[xi - n:xi, n:yi - n] = w1b
        G_1[xi - n:xi, n:yi - n] = w1g
        R_1[xi - n:xi, n:yi - n] = w1r

        # left
        B_1[n:xi - n, 0:n] = w4b
        G_1[n:xi - n, 0:n] = w4g
        R_1[n:xi - n, 0:n] = w4r

        # right
        B_1[n:xi - n, yi - n:yi] = w3b
        G_1[n:xi - n, yi - n:yi] = w3g
        R_1[n:xi - n, yi - n:yi] = w3r

        # top left
        B_1[0:n, 0:n] = w2lb
        G_1[0:n, 0:n] = w2lg
        R_1[0:n, 0:n] = w2lr

        # botton right
        B_1[xi - n:xi, yi - n:yi] = w1rb
        G_1[xi - n:xi, yi - n:yi] = w1rg
        R_1[xi - n:xi, yi - n:yi] = w1rr

        # top right
        B_1[0:n, yi - n:yi] = w2rb
        G_1[0:n, yi - n:yi] = w2rg
        R_1[0:n, yi - n:yi] = w2rr

        # botton left
        B_1[xi - n:xi, 0:n] = w1lb
        G_1[xi - n:xi, 0:n] = w1lg
        R_1[xi - n:xi, 0:n] = w1lr

        rx = len(B_1)
        ry = len(B_1[0])
        B_1[n:rx - n, n:ry - n] = B
        G_1[n:rx - n, n:ry - n] = G
        R_1[n:rx - n, n:ry - n] = R
        new_img_cv = np.dstack((B_1, G_1, R_1))
        return new_img_cv

    if inp == 2:
        n = padsize
        sizex = len(img_grey)
        sizey = len(img_grey[0])
        new_img = np.zeros((sizex + 2 * n, sizey + 2 * n))
        x = img_grey.shape[0]
        y = img_grey.shape[1]
        xi = new_img.shape[0]
        yi = new_img.shape[1]

        # Bottom
        w1b = img_grey[x - n - 1:x - 1, :]
        w1b = w1b[::-1]

        # Top
        w2b = img_grey[1:n + 1, :]
        w2b = w2b[::-1]

        # Right
        w3b = img_grey[:, y - n - 1:y - 1]
        w3b = np.fliplr(w3b)

        # Left
        w4b = img_grey[:, 1:n + 1]
        w4b = np.fliplr(w4b)

        # Top left
        w2lb = w2b[:, 1:n + 1]
        w2lb = np.fliplr(w2lb)

        # Top Right
        w2rb = w2b[:, y - n - 1:y - 1]
        w2rb = np.fliplr(w2rb)

        # Bottom left
        w1lb = w1b[:, 1:n + 1]
        w1lb = np.fliplr(w1lb)

        # Bottom right
        w1rb = w1b[:, y - n - 1:y - 1]
        w1rb = np.fliplr(w1rb)

        # Placing on new matrix
        # top
        new_img[0:n, n:yi - n] = w2b

        # botton
        new_img[xi - n:xi, n:yi - n] = w1b

        # left
        new_img[n:xi - n, 0:n] = w4b

        # right
        new_img[n:xi - n, yi - n:yi] = w3b

        # top left
        new_img[0:n, 0:n] = w2lb

        # botton right
        new_img[xi - n:xi, yi - n:yi] = w1rb

        # top right
        new_img[0:n, yi - n:yi] = w2rb

        # botton left
        new_img[xi - n:xi, 0:n] = w1lb

        rx = len(new_img)
        ry = len(new_img[0])
        new_img[n:rx - n, n:ry - n] = img_grey

        return new_img


def zeroPad(padsize, B, G, R, img_grey, inp):
    # RGB
    if inp == 1:

        n = padsize
        sizex = len(B)
        sizey = len(B[0])
        new_img = np.zeros((sizex + 2 * n, sizey + 2 * n, 3))
        B_1 = new_img[:, :, 0]
        G_1 = new_img[:, :, 1]
        R_1 = new_img[:, :, 2]
        rx = len(B_1)
        ry = len(B_1[0])
        B_1[n:rx - n, n:ry - n] = B
        G_1[n:rx - n, n:ry - n] = G
        R_1[n:rx - n, n:ry - n] = R
        new_img_cv = np.dstack((B_1, G_1, R_1))

    # Gray Scale
    elif inp == 2:

        n = padsize
        sizex = len(img_grey)
        sizey = len(img_grey[0])
        new_img_cv = np.zeros((sizex + 2 * n, sizey + 2 * n))
        rx = len(new_img_cv)
        ry = len(new_img_cv[0])
        new_img_cv[n:rx - n, n:ry - n] = img_grey
    return new_img_cv


def convolution(padsize, box, new_img_cv, inp):
    if inp == 1:
        B_1 = new_img_cv[:, :, 0]
        G_1 = new_img_cv[:, :, 1]
        R_1 = new_img_cv[:, :, 2]

        pixelx = 0
        pixely = 0

        strides = 1;
        bx = box.shape[0]
        by = box.shape[1]

        if box.shape[0] % 2 == 0:
            for i in range(B_1.shape[0]):
                added = i + bx
                if added < B_1.shape[0]:
                    pixelx += 1
            for i in range(B_1.shape[1]):
                added = i + by
                if added < B_1.shape[1]:
                    pixely += 1

        else:
            for i in range(B_1.shape[0]):
                added = i + bx
                if added <= B_1.shape[0]:
                    pixelx += 1
            for i in range(B_1.shape[1]):
                added = i + by
                if added <= B_1.shape[1]:
                    pixely += 1

        k = box.shape[0]
        convolved_img_B = np.zeros(shape=(pixelx, pixely)).astype(np.float32)
        convolved_img_G = np.zeros(shape=(pixelx, pixely)).astype(np.float32)
        convolved_img_R = np.zeros(shape=(pixelx, pixely)).astype(np.float32)

        for i in range(pixelx):
            for j in range(pixely):
                conlv = B_1[i:i + k, j:j + k]
                s = (np.sum(np.multiply(conlv, box))).astype(np.float32)
                if s > 255:
                    convolved_img_B[i][j] = 255
                elif s < 0:
                    convolved_img_B[i][j] = 0
                else:
                    convolved_img_B[i][j] = s

        for i in range(pixelx):
            for j in range(pixely):
                conlv = G_1[i:i + k, j:j + k]
                s = (np.sum(np.multiply(conlv, box))).astype(np.float32)
                if s > 255:
                    convolved_img_G[i][j] = 255
                elif s < 0:
                    convolved_img_G[i][j] = 0
                else:
                    convolved_img_G[i][j] = s

        for i in range(pixelx):
            for j in range(pixely):
                conlv = R_1[i:i + k, j:j + k]
                s = (np.sum(np.multiply(conlv, box))).astype(np.float32)
                if s > 255:
                    convolved_img_R[i][j] = 255
                elif s < 0:
                    convolved_img_R[i][j] = 0
                else:
                    convolved_img_R[i][j] = s
        new_img_cv = np.dstack((convolved_img_B, convolved_img_G, convolved_img_R))
        return new_img_cv

    elif inp == 2:

        pixelx = 0
        pixely = 0

        strides = 1;
        bx = box.shape[0]
        by = box.shape[1]

        if box.shape[0] % 2 == 0:
            for i in range(new_img_cv.shape[0]):
                added = i + bx
                if added < new_img_cv.shape[0]:
                    pixelx += 1
            for i in range(new_img_cv.shape[1]):
                added = i + by
                if added < new_img_cv.shape[1]:
                    pixely += 1

        else:
            for i in range(new_img_cv.shape[0]):
                added = i + bx
                if added <= new_img_cv.shape[0]:
                    pixelx += 1
            for i in range(new_img_cv.shape[1]):
                added = i + by
                if added <= new_img_cv.shape[1]:
                    pixely += 1

        # print('p', pixelx, ' ',pixely)
        k = box.shape[0]
        convolved_img_B = np.zeros(shape=(pixelx, pixely)).astype(np.float32)

        # box = box/9;
        for i in range(pixelx):
            for j in range(pixely):
                conlv = new_img_cv[i:i + k, j:j + k]
                s = (np.sum(np.multiply(conlv, box))).astype(np.float32)
                if s > 255:
                    convolved_img_B[i][j] = 255
                elif s < 0:
                    convolved_img_B[i][j] = 0
                else:
                    convolved_img_B[i][j] = s

        return convolved_img_B


def conv2(img, box):
    pads = padSize(box)
    print("image sampling size = ", img.shape)
    if (len(img.shape) < 3):
        inp = 2
        img_grey = img

    else:
        inp = 1
        B = img[:, :, 0]
        G = img[:, :, 1]
        R = img[:, :, 2]
    pad = "clip/zero-padding"
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if (pad == "clip/zero-padding"):
        new_img_cv = zeroPad(pads, B, G, R, img_grey, inp)

    elif (pad == "wrap around"):
        new_img_cv = wraparound(pads, B, G, R, img_grey, inp)

    elif (pad == "copy edge"):
        new_img_cv = copyedge(pads, B, G, R, img_grey, inp)

    elif (pad == "reflect across edge"):
        new_img_cv = reflect(pads, B, G, R, img_grey, inp)
    new_img_cv2 = convolution(pads, box, new_img_cv, inp)
    new_img_cv2 = np.float32(new_img_cv2)
    return new_img_cv2

