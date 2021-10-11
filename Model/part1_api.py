try:
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter

    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


def find_tfl_lights(c_image: np.ndarray, **kwargs):
    """
    Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
    :param c_image: The image itself as np.uint8, shape of (H, W, 3)
    :param kwargs: Whatever config you want to pass in here
    :return: 4-tuple of x_red, y_red, x_green, y_green
    """
    imgRed = Image.fromarray(c_image[:, :, 0])
    imgGreen = Image.fromarray(c_image[:, :, 1])
    # plt.imshow(imgRed)
    # plt.show(block=True)

    # kernel_3 = np.array([[0, 0.01, 0],
    #                     [0.01, 0.96, 0.01],
    #                     [0, 0.01, 0]])

    kernel_5 = np.array([[-0.065, -0.065, -0.065, -0.065, -0.065],
                         [-0.065,  0.05,   0.4,    0.05,  -0.065],
                         [-0.065,  0.4,    0.4,    0.4,   -0.065],
                         [-0.065,  0.05,   0.4,    0.05,  -0.065],
                         [-0.065, -0.065, -0.065, -0.065, -0.065]])

    red_conv = sg.convolve(imgRed, kernel_5, mode='same')
    green_conv = sg.convolve(imgGreen, kernel_5, mode='same')

    red_image_conv = Image.fromarray(red_conv)
    green_image_conv = Image.fromarray(green_conv)

    # plt.imshow(red_image_conv)
    # plt.imshow(green_image_conv)
    # plt.show(block=True)

    red_max_flt = maximum_filter(red_image_conv, size=350)
    green_max_flt = maximum_filter(green_image_conv, size=350)
    # plt.imshow(Image.fromarray(red_max_flt))
    # plt.show(block=True)
    # plt.imshow(Image.fromarray(green_max_flt))
    # plt.show(block=True)

    red_points = np.subtract(red_max_flt, red_image_conv)
    green_points = np.subtract(green_max_flt, green_image_conv)
    # plt.imshow(Image.fromarray(red_points))
    # plt.show(block=True)

    # red_same_points = np.argwhere(red_max_flt > 230)
    # green_same_points = np.argwhere(green_max_flt > 230)
    red_same_points = np.argwhere(red_points == 0)
    green_same_points = np.argwhere(green_points == 0)

    return red_same_points[:, 1], red_same_points[:, 0], green_same_points[:, 1], green_same_points[:, 0]


# GIVEN CODE TO TEST YOUR IMPLENTATION AND PLOT THE PICTURES
def show_image_and_gt(image, objs, fig_num=None):
    plt.figure(fig_num).clf()
    plt.imshow(image)
    labels = set()
    if objs is not None:
        for o in objs:
            poly = np.array(o['polygon'])[list(np.arange(len(o['polygon']))) + [0]]
            plt.plot(poly[:, 0], poly[:, 1], 'r', label=o['label'])
            labels.add(o['label'])
        if len(labels) > 1:
            plt.legend()


def test_find_tfl_lights(image_path, json_path=None, fig_num=None):
    """
    Run the attention code
    """
    image = np.array(Image.open(image_path))
    if json_path is None:
        objects = None
    else:
        gt_data = json.load(open(json_path))
        what = ['traffic light']
        objects = [o for o in gt_data['objects'] if o['label'] in what]

    show_image_and_gt(image, objects, fig_num)

    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    plt.plot(red_x, red_y, 'rx', markersize=4)
    plt.plot(green_x, green_y, 'g+', markersize=4)


def main(argv=None):
    """It's nice to have a standalone tester for the algorithm.
    Consider looping over some images from here, so you can manually exmine the results
    Keep this functionality even after you have all system running, because you sometime want to debug/improve a module
    :param argv: In case you want to programmatically run this"""

    parser = argparse.ArgumentParser("Test TFL attention mechanism")
    parser.add_argument('-i', '--image', type=str, help='Path to an image')
    parser.add_argument("-j", "--json", type=str, help="Path to json GT for comparison")
    parser.add_argument('-d', '--dir', type=str, help='Directory to scan images in')
    args = parser.parse_args(argv)
    default_base = r'C:\Users\estri\PycharmProjects\traffic_mobileye\aachen'

    if args.dir is None:
        args.dir = default_base
    flist = glob.glob(os.path.join(args.dir, '*_leftImg8bit.png'))

    for image in flist:
        json_fn = image.replace('_leftImg8bit.png', '_gtFine_polygons.json')

        if not os.path.exists(json_fn):
            json_fn = None
        test_find_tfl_lights(image, json_fn)

    if len(flist):
        print("You should now see some images, with the ground truth marked on them. Close all to quit.")
    else:
        print("Bad configuration?? Didn't find any picture to show")
    plt.show(block=True)


if __name__ == '__main__':
    main()
