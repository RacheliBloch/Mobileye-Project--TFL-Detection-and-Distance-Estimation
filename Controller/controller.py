from Model.TFL_Manager import manage

def run():
    play_list = r'C:\Users\estri\PycharmProjects\traffic_mobileye\Controller\play_list.pls'
    ply_lst_file = open(play_list)
    pkl_path = ply_lst_file.readline().replace("\n", '')
    prev_frame_id = int(ply_lst_file.readline().replace("\n", ''))

    abs_path = r"C:\\Users\\estri\\PycharmProjects\\traffic_mobileye\\"
    prev_img_path = ply_lst_file.readline().replace("\n", '').replace('//', r"\\")
    prev_img_path = abs_path + prev_img_path
    while True:
        curr_img_path_1 = ply_lst_file.readline().replace("\n", '').replace('//', r"\\")
        curr_img_path = abs_path + curr_img_path_1

        if not curr_img_path_1:
            break

        manage(prev_img_path,curr_img_path, abs_path + pkl_path, prev_frame_id)
        prev_frame_id += 1
        prev_img_path = curr_img_path

# run()