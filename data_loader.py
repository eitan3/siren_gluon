import numpy as np
import mxnet as mx
import random
import copy
import cv2
from utils import create_x_input, norm_coord_to_real


class DataIter(mx.io.DataIter):
    def __init__(self, ctx, batch_size, filename):
        super(DataIter, self).__init__()
        self.ctx = ctx
        self.batch_size = batch_size

        data_input = cv2.imread(filename)
        self.img_shape = data_input.shape
        x = create_x_input(self.img_shape)
        y = data_input.reshape(-1, 3)

        for coord, pixel_value in zip(x, y):
            c = norm_coord_to_real(coord, self.img_shape)
            assert (data_input[c[0], c[1], :] == pixel_value).all(), "Pixel values do not match"

        self.data_shapes = (batch_size, 2)
        self.label_shapes = (batch_size, 3)
        self.examples = []
        y = y / 255.0
        for i in range(x.shape[0]):
            exmp = {'input': x[i],
                    'output': y[i]}
            self.examples.append(exmp)
        self.num_examples = len(self.examples)
        self.num_batches = np.floor(self.num_examples / self.batch_size)

        self.cur_batch = 0
        self.reset()

    def __iter__(self):
        return self

    def reset(self):
        random.shuffle(self.examples)
        self.cur_batch = 0

    def __next__(self):
        return self.next()

    def norm_coord_to_real(self, coord):
        image_shape_t = np.array([self.img_shape[0], self.img_shape[1]])
        c = (image_shape_t * 0.5 * (1 + coord))
        c = [int(round(c[0])), int(round(c[1]))]
        return c

    def next(self):
        if self.cur_batch < self.num_batches:
            data = np.zeros(self.data_shapes, dtype=np.float32)
            label = np.zeros(self.label_shapes, dtype=np.float32)

            for i in range(self.batch_size):
                example_i = copy.deepcopy(self.examples[self.cur_batch * self.batch_size + i])
                img = example_i['input']
                lbl = example_i['output']

                data[i, :] = img[:]
                label[i, :] = lbl[:]
                del img
                del lbl

            self.cur_batch += 1
            return mx.nd.array(data, ctx=self.ctx), mx.nd.array(label, ctx=self.ctx)
        else:
            self.reset()
            raise StopIteration
