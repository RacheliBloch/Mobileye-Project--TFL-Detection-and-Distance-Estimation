import numpy as np, matplotlib.pyplot as plt
from os.path import join


def load_tfl_data(data_dir, crop_shape=(81, 81)):
    images = np.memmap(join(data_dir, 'data.bin'), mode='r', dtype=np.uint8).reshape([-1] + list(crop_shape) + [3])
    labels = np.memmap(join(data_dir, 'labels.bin'), mode='r', dtype=np.uint8)
    return {'images': images, 'labels': labels}


def viz_my_data(images, labels, predictions=None, num=(5, 5), labels2name={0: 'No TFL', 1: 'Yes TFL'}):
    assert images.shape[0] == labels.shape[0]
    assert predictions is None or predictions.shape[0] == images.shape[0]
    h = 5
    n = num[0] * num[1]
    ax = plt.subplots(num[0], num[1], figsize=(h * num[0], h * num[1]), gridspec_kw={'wspace': 0.05}, squeeze=False,
                      sharex=True, sharey=True)[1]  # .flatten()
    idxs = np.random.randint(0, images.shape[0], n)
    for i, idx in enumerate(idxs):
        ax.flatten()[i].imshow(images[idx])
        title = labels2name[labels[idx]]
        if predictions is not None: title += ' Prediction: {:.2f}'.format(predictions[idx])
        ax.flatten()[i].set_title(title)
    plt.show(block=True)


# root = './'  #this is the root for your val and train datasets
data_dir = r'C:\Users\estri\PycharmProjects\traffic_mobileye\CityScapes\gtFine'
datasets = {
    'val': load_tfl_data(join(data_dir, 'val')),
    'train': load_tfl_data(join(data_dir, 'train')),
}
for k, v in datasets.items():
    print('{} :  {} 0/1 split {:.1f} %'.format(k, v['images'].shape, np.mean(v['labels'] == 1) * 100))

viz_my_data(num=(6, 6), **datasets['val'])
