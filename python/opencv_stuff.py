import cv2,time
import numpy as np

img = cv2.imread('/home/rick/Pictures/wallpaperflare.com_wallpaper (2).jpg')
# find what second argument does!
#img = cv2.imread('/home/rick/Pictures/wallpaperflare.com_wallpaper (2).jpg',cv2.IMREAD_GRAYSCALE)

img2 = np.zeros((500,500,3),dtype='uint8')

if __name__ == '__main__':
    print(f'Width: {img.shape[1]} pixels')
    print(f'Height: {img.shape[0]} rows')
    print(f'Channels: {img.shape[2]}')
    blue = (255, 0, 0)

    red = (0, 0, 255)

    # filter an image slicing it and setting a channel to 0
    img[:,:,0] = 0

    # scale an image
    upscaled_img = cv2.resize(img, (img.shape[0] * 2, img.shape[1] * 2))
    downscaled_img = cv2.resize(img, (img.shape[0] // 3, img.shape[1] //3))
    # draw a line
    cv2.line(img2, (10, 10), (200, 200), (255, 0, 0), 50)

    # draw a rectangle
    cv2.rectangle(img2, (10, 30), (200, 200), red, 10)

    # draw a circle
    centerX, centerY = (img2.shape[0] // 2),  (img2.shape[1] // 2)
    print(centerX, centerY, sep = ' ')
    cv2.circle(img2, (centerX, centerY), 30, red, 10)


    # THE CHANNELS ARE SWAPPED! opencv reads the images in this format
    b,g,r = img[0,0]

    print(b,g,r, sep = ' ')
    # write an image to a specific format
    #cv2.imwrite('test.jpeg', img)

    # pick a corner of the image
    corner = img[0:200, 0:200]

    # turn a corner of the image to a specific color
    img[0:200, 0:200] = (0,0,0)

    # apply a transformation to an image
    M = np.float32(((1,0,50), (0,1,100)))
    rows, cols = img.shape[0], img.shape[1]
    dst_image = cv2.warpAffine(img, M, (rows,cols))

    # rotate an image
    M = cv2.getRotationMatrix2D((cols // 2, rows // 2), 180, 1)
    dst_image = cv2.warpAffine(img, M, (rows*2, cols*2))

    # not sure about what this does
    pts_1 = np.float32([[135,45], [385,45], [135,230]])
    pts_2 = np.float32([[135,45], [385,45], [150,230]])

    dst_image = cv2.warpAffine(pts_1, M, (800,800))

    M = cv2.getAffineTransform(pts_1, pts_2)
    # Show the image. waitKey method takes as input the amount of milliseconds you want to show the image. 0 stands for "open until closed"
    cv2.imshow('Image', dst_image)

    # close the image by pressing any key
    cv2.waitKey(0)
