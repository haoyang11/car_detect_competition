import cv2
import sys
import os
import json
import math
import pickle
import numpy as np
#from utils.timer import Timer
from projection import ground_point_to_bird_view_proj as gp2bv
from projection import bird_view_proj_to_ground_point as bv2gp
#import natsort

"""
pickle example

a = {'hello': 'world'}

with open('grid.pkl', 'wb') as handle:
    pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('grid.pkl', 'rb') as handle:
    b = pickle.load(handle)

print a == b
"""

raw_image_path = ""

grid_cache_path = "grid_x50_y3.pkl"


def draw(image, x, y, grid_cache, display_photo_coordinate=False):
    if (x, y) not in grid_cache:
        grid_cache[(x, y)] = bv2gp(x, y, cam_param)
    u, v = grid_cache[(x, y)]
    grid_cache[(x, y)] = u, v
    text = str(x) + "," + str(y)
    if display_photo_coordinate:
        text += " " + str(u) + "," + str(v)
    cv2.putText(image, text, (u, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    cv2.circle(image, (u, v), 3, (0, (x + y + 103) % 2 * 255, (x + y + 102) % 2 * 255), -1)


def draw_image(path, image_file_name, grid_cache):
    print("drawing in image : ", image_file_name)
    image = cv2.imread(os.path.join(path, image_file_name))
    #cv2.imwrite('output/'+str(tot)+".jpg", image)

    for x in range(5, 50, 3):
        for y in range(-3, 4):
            draw(image, x, y, grid_cache)
        a = grid_cache[(x, 2)]
        b = grid_cache[(x, -2)]
        alpha = math.atan2(b[1] - a[1], b[0] - a[0])
        tan_alpha = (b[1] - a[1]) / (b[0] - a[0])
        c = grid_cache[(x, -1)]
        #print("x = ", x, ", aph = " + str(round(alpha, 3)) + ", tan(aph) = " + str(round(tan_alpha, 5)))
        cv2.putText(image,
            "aph = " + str(round(alpha, 3)) + ", tan(aph) = " + str(round(tan_alpha, 5)),
            (c[0] + 20, c[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 120, 255), 2)

    for x in range(200, 1000, 600):
        for y in range(0, 1, 1):
            draw(image, x, y, grid_cache, display_photo_coordinate=True)
    #draw(300, 0, grid_cache)
    #draw(400, 0, grid_cache)
    cv2.imwrite(os.path.join(path, "grided_" + image_file_name), image)
    #cv2.imwrite('output_lyc/'+ str(nb) + '_' + str(tot)+".jpg", image)


def draw_image_in_path(path, grid_cache):
    for file_name in os.listdir(path):
    #for file_name in natsort.natsorted(os.listdir(path)):
        if file_name.find("grided") < 0 and file_name.split(".")[-1] == "jpg":
            draw_image(path, file_name, grid_cache)


"""usage :

    python drawgrid image_path

    python drawgrid image_1.jpg image_2.jpg image_3.jpg
"""
if __name__ == '__main__':
    with open("camera_parameter.json") as f:
        cam_param = json.load(f)

    grid_cache = {}
    try:
        print("loading grid cache from ", grid_cache_path)
        with open(grid_cache_path, 'rb') as handle:
            grid_cache = pickle.load(handle)
    except IOError:
        print("Error: 没有找到文件或读取文件失败")
    else:
        print("grid cache loaded")

    #timer_tot = Timer()
    #timer_tot.tic()

    if len(sys.argv) == 1:
        #raise NotImplementedError
        draw_image_in_path(".", grid_cache)
        exit(0)



    if os.path.isdir(sys.argv[1]):
        draw_image_in_path(sys.argv[1], grid_cache)
    else:
        for file_name in sys.argv[1:]:
            if os.path.isdir(file_name):
                draw_image_in_path(file_name, grid_cache)
            else:
                draw_image(file_name, grid_cache)

    print("saving grid cache to ", grid_cache_path)
    with open(grid_cache_path, 'wb') as handle:
        pickle.dump(grid_cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    #timer_tot.toc()
    #print(timer_tot.total_time)
