import seaborn as sbn
import numpy as np
from part_2_train import val, m
from part_2_validate import viz_my_data

predictions = m.predict(val['images'])
sbn.distplot(predictions[:, 0])

predicted_label = np.argmax(predictions, axis=-1)
print('accuracy:', np.mean(predicted_label == val['labels']))

viz_my_data(num=(6, 6), predictions=predictions[:, 1], **val)
