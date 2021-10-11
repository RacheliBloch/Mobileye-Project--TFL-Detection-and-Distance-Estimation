import Point as Point
from PIL import Image
import os
import json
import glob
import numpy as np
from os.path import join

from matplotlib.patches import Polygon

from part1_api import find_tfl_lights


def store_data(path):
    data_path = os.path.join(path, 'data.bin')
    # if not os.path.exists(data_path):
    data_file = open(data_path, "ab")

    labels_path = os.path.join(path, 'labels.bin')
    # if not os.path.exists(labels_path):
    labels_file = open(labels_path, "ab")

    for dir in os.listdir(path):
        json_files = glob.glob(os.path.join(path + '\\' + dir, '*.json'))
        for js in json_files:
            f = open(js)
            objects = json.load(f)['objects']
            for o in objects:
                if o['label'] == 'traffic light':
                    # data = open(
                    #     os.path.join(data_path, os.path.basename(js).replace('gtFine_polygons.json', 'data1.bin')), 'w')
                    # labels = open(
                    #     os.path.join(labels_path, os.path.basename(js).replace('gtFine_polygons.json', 'labels1.bin')),
                    #     'w')
                    x_center, y_center = centroid(o['polygon'])
                    original = js.replace('_gtFine_polygons.json', '_leftImg8bit.png')
                    original = original.replace('gtFine', 'leftImg8bit')
                    im = Image.open(original)
                    cropped_im = im.crop((x_center - 40, y_center - 40, x_center + 41, y_center + 41))

                    data_array = np.asarray(cropped_im)
                    np.ndarray.tofile(data_array, data_file)

                    labels_file.write(bytearray([1]))
                    a = np.asarray(im)
                    red_x, red_y, green_x, green_y = find_tfl_lights(a)
                    red_points = list(zip(red_x, red_y))
                    green_points = list(zip(green_x, green_y))
                    points = red_points + green_points

                    flag = False
                    for k in objects:
                        if k['label'] != 'traffic light':
                            polygon = Polygon(k['polygon'])
                            for point in points:
                                point1 = Point(point[0], point[1])
                                if polygon.contains(point1):
                                    x_center, y_center = point[0], point[1]
                                    # original = js.replace('_gtFine_polygons.json', '_leftImg8bit.png')
                                    # original = original.replace('gtFine', 'leftImg8bit')
                                    # im = Image.open(original)
                                    cropped_im = im.crop((x_center - 40, y_center - 40, x_center + 41, y_center + 41))

                                    data_array = np.asarray(cropped_im)
                                    np.ndarray.tofile(data_array, data_file)

                                    labels_file.write(bytearray([0]))
                                    flag = True
                                    break
                            if flag:
                                break
                    if not flag:
                        for k in objects:
                            if k['label'] != 'traffic light':
                                x_center, y_center = centroid(k["polygon"])
                                # original = js.replace('_gtFine_polygons.json', '_leftImg8bit.png')
                                # original = original.replace('gtFine', 'leftImg8bit')
                                # im = Image.open(original)
                                cropped_im = im.crop((x_center - 40, y_center - 40, x_center + 41, y_center + 41))

                                data_array = np.asarray(cropped_im)
                                np.ndarray.tofile(data_array, data_file)

                                labels_file.write(bytearray([0]))
                                break


                    break

    data_file.close()
    labels_file.close()


def centroid(vertexes):
    _x_list = [vertex[0] for vertex in vertexes]
    _y_list = [vertex[1] for vertex in vertexes]
    _len = len(vertexes)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return round(_x), round(_y)


if __name__ == '__main__':
    path = r"C:\Users\User\Desktop\Excelenteam\mobileye\gtFine"
    store_data(join(path, 'val'))
    store_data(join(path, 'train'))
