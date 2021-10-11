import numpy as np
from PIL import Image

from Model import SFM
from Model.SFM_standAlone import init_SFM
from Model.check_point import check_tfl
from Model.part1_api import find_tfl_lights


def part1(img_path):
    image = np.array(Image.open(img_path))
    red_x, red_y, green_x, green_y = find_tfl_lights(image)
    red_points = list(zip(red_x, red_y))
    green_points = list(zip(green_x, green_y))
    return red_points + green_points


def part2(img_path, suspicious_points):
    return check_tfl(img_path, suspicious_points)


def part3(prev_img_path, curr_img_path, pkl_path, prev_tfl, curr_tfl, prev_frame_id):
    # SFM.calc_TFL_dist(prev_container, curr_container, focal, pp)
    init_SFM(prev_img_path, curr_img_path, pkl_path, prev_tfl, curr_tfl, prev_frame_id)


#
#
# def run():
#     play_list = r'C:\Users\estri\PycharmProjects\traffic_mobileye\Controller\play_list.pls'
#     ply_lst_file = open(play_list)
#     pkl_path = ply_lst_file.readline().replace("\n", '')
#
#     prev_frame_id = int(ply_lst_file.readline().replace("\n", ''))
#
#     while True:
#         abs_path = r"C:\\Users\\estri\\PycharmProjects\\traffic_mobileye\\"
#         prev_img_path = ply_lst_file.readline().replace("\n", '').replace('//', r"\\")
#         prev_img_path = abs_path + prev_img_path
#         curr_img_path_1 = ply_lst_file.readline().replace("\n", '').replace('//', r"\\")
#         curr_img_path = abs_path + curr_img_path_1
#
#         if not curr_img_path_1:
#             break
#         sus_prev_tfl = part1(prev_img_path)
#         sus_curr_tfl = part1(curr_img_path)
#
#         prev_tfl = part2(prev_img_path, sus_prev_tfl)
#         curr_tfl = part2(curr_img_path, sus_curr_tfl)
#
#         init_SFM(prev_img_path, curr_img_path, abs_path + pkl_path, prev_tfl, curr_tfl, prev_frame_id)
#         prev_frame_id += 2

def manage(prev_img_path, curr_img_path, pkl_path, prev_frame_id):
    sus_prev_tfl = part1(prev_img_path)
    sus_curr_tfl = part1(curr_img_path)

    prev_tfl = part2(prev_img_path, sus_prev_tfl)
    curr_tfl = part2(curr_img_path, sus_curr_tfl)

    part3(prev_img_path, curr_img_path, pkl_path, prev_tfl, curr_tfl, prev_frame_id)
