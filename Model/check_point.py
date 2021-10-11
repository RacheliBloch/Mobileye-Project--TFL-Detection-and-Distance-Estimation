import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

loaded_model = load_model(r"C:\Users\estri\PycharmProjects\traffic_mobileye\Model\model.h5")


def check_tfl(img_path, suspicious_points):
    im = Image.open(img_path)
    # plt.imshow(im)
    # plt.show(block=True)

    tfl_result = []
    for pt in suspicious_points:
        crop_im = np.asarray(im.crop((pt[0] - 40, pt[1] - 40, pt[0] + 41, pt[1] + 41)))
        # plt.imshow(crop_im)
        # plt.show(block=True)
        pred = loaded_model.predict(np.asarray([crop_im]))
        print("pred: ", pred[0][0])
        if pred[0][1] > 0.70:
            tfl_result.append(pt)

    im.close()
    return tfl_result


