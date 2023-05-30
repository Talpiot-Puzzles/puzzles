import PIL.Image
import numpy
from PIL import Image, ImageChops

H1 = 30 # the hight of the drown relative to the trees serfice
H2 = 5 # the hight of the trees
n = 640 # number of pixels in the long axix of the image
tet = 57 # the opaning angle of the long axix of the image
METERS_PER_PIXEL = 2*H1*numpy.tan(tet*numpy.pi/360)/n
#METERS_PER_PIXEL = 0.00118

def refocuse_two_images (img: PIL.Image.Image, dx, dy, refocused_image_name,h1 = H1, dh = H2):
    """

    :param img: the second image (moves). the first image is not touched, therfore irrelevant to the function
    :param dx: the x movement (by meters) between the two images
    :param dy: the y movement (by meters) between the two images
    :param dh: the depth of the refocuse
    :return:
    """
    l = numpy.sqrt(numpy.power(dx, 2) + numpy.power(dy, 2))
    refocuse_length_meters = l*dh/(dh+h1)
    refocuse_length_pixels = refocuse_length_meters/METERS_PER_PIXEL

    refocuse_x_meters = -refocuse_length_meters*dx/l
    refocuse_y_meters = -refocuse_length_meters*dy/l


    refocuse_x_pixels = int(round(refocuse_length_pixels*dx/l)*numpy.sign(dx))
    refocuse_y_pixels = -int(round(refocuse_length_pixels*dy/l)*numpy.sign((dy)))

    img_shifted =  ImageChops.offset(img,refocuse_x_pixels,refocuse_y_pixels)
    img_shifted.save(refocused_image_name)
    return img_shifted



def refocuse_multipule_images(shift_list, h1, h2):
    """
    :param shift_list: a list of taples that contain the relative shift between the photo and a mother photo in order to focuse on h1 plain.
    the shift is represented in pixels
    :param h1: the hight of the original plain (old focuse)
    :param h2: the hight of the new plain (new focuse)
    :return: the shift list in order to focuse on h2
    """

    dh = h2-h1
    new_shift_lst = []
    for tup in shift_list:
        (dx, dy) = tup
        refocuse_x_pixels = -dh*dx/h1
        refocuse_y_pixels = -dh*dy/h1
        new_shift_lst.append((dx+refocuse_x_pixels, dy+refocuse_y_pixels))
    return new_shift_lst


def get_spatial_engle(x,y, X = 640, Y = 360, HFOV = 57):
    """

    :param x: the x location of the pixel
    :param y: the y location of the pixel
    :param X: the X length of the picture
    :param Y: the Y length of the picture
    :param tet: HFOV (57 degrees in mavic's thermal camera)
    :return:
    """
    rad_HFOV = HFOV*numpy.pi/180
    xm = x - X
    ym = y - Y
    tantet = numpy.sqrt(numpy.power(xm,2) + numpy.power(ym,2))/X*numpy.tan(rad_HFOV)
    tet = numpy.arctan(tantet)
    phi = numpy.arctan2(ym, xm)
    return (tet, phi)

# print (get_spatial_engle(1092, 812)[0]*180/numpy.pi, get_spatial_engle(1092, 812)[1]*180/numpy.pi)