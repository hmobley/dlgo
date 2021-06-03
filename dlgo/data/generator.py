import glob
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
class DataGenerator:
    def __init__(self, data_directory, samples):
        print("DataGenerator.__init__ called")
        self.data_directory = data_directory
        self.samples = samples
        self.files = set(file_name for file_name, index in samples)
        self.num_samples = None

    def get_num_samples(self, batch_size=128, num_classes=19*19):
        print("DataGenerator.get_num_samples called")
        if self.num_samples is not None:
            return self.num_samples
        else:
            self.num_samples = 0
            for X, y in self._generate(batch_size=batch_size,
                                       num_classes=num_classes):
                self.num_samples += X.shape[0]
            return self.num_samples

    def _generate(self, batch_size, num_classes):
        print("_generate called")
        czip = 0
        cfeature = 0
        cbatch = 0
        for zip_file_name in self.files:
            czip += 1
            file_name = zip_file_name.replace('.tar.gz', '') + 'train'
            base = self.data_directory = '/' + file_name + '_features_*.npy'
            for feature_file in glob.glob(base):
                cfeature += 1
                label_file = feature_file.replace('features', 'labels')
                x = np.load(feature_file)
                y = np.load(label_file)
                x = x.astype('float32')
                y = to_categorical(y.astype(int), num_classes)
                while x.shape[0] >= batch_size:
                    cbatch += 1
                    x_batch, x = x[:batch_size], x[batch_size:]
                    y_batch, y = y[:batch_size], y[batch_size:]
                    print("yield {}, {}, {}".format(czip,cfeature,cbatch))
                    yield x_batch, y_batch

    def generate(self, batch_size=128, num_classes=19 * 19):
        print("generate called")
        outer = 0
        inner = 0
        while True:
            outer += 1
            for item in self._generate(batch_size, num_classes):
                inner += 1
                print("yield counts: outer: {}, inner: {}".format(outer,inner))
                yield item

    
