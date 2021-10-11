from part_2_validate import load_tfl_data
from part_2_model_training import tfl_model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.optimizers import Adam
from os.path import join
import matplotlib.pyplot as plt

data_dir = r'C:\Users\estri\PycharmProjects\traffic_mobileye\CityScapes\gtFine'
datasets = {
    'val': load_tfl_data(join(data_dir, 'val')),
    'train': load_tfl_data(join(data_dir, 'train')),
}
# prepare our model
m = tfl_model()
m.compile(optimizer=Adam(), loss=sparse_categorical_crossentropy, metrics=['accuracy'])

train, val = datasets['train'], datasets['val']
# train it, the model uses the 'train' dataset for learning. We evaluate the "goodness" of the model, by predicting the label of the images in the val dataset.
history = m.fit(train['images'], train['labels'], validation_data=(val['images'], val['labels']), epochs=8)

# compare train vs val acccuracy,
# why is val_accuracy not as good as train accuracy? are we overfitting?
epochs = history.history
epochs['train_acc'] = epochs['accuracy']
plt.figure(figsize=(10, 10))
for k in ['train_acc', 'val_accuracy']:
    plt.plot(range(len(epochs[k])), epochs[k], label=k)

plt.legend();
plt.show(block=True)
