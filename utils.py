import numpy as np
import mxnet as mx

def create_x_input(img_shape):
    g0, g1 = np.meshgrid(np.arange(0, img_shape[1], step=1.0), np.arange(0, img_shape[0], step=1.0))
    x = np.stack((g1, g0), axis=2)
    x = x.reshape(-1, 2).astype(dtype=np.float32)
    x[:, 0] = (x[:, 0] / img_shape[0]) * 2.0 - 1.0  # Height
    x[:, 1] = (x[:, 1] / img_shape[1]) * 2.0 - 1.0  # Width
    return x


def norm_coord_to_real(coord, img_shape):
    image_shape_t = np.array([img_shape[0], img_shape[1]])
    c = (image_shape_t * 0.5 * (1 + coord))
    c = [int(round(c[0])), int(round(c[1]))]
    return c


def evaluate_accuracy(data_iterator, net, loss_function, ctx):
    acc = mx.metric.MSE()
    loss = 0
    num_batches = 0
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        out = net(data)
        loss += loss_function(out, label).mean().asscalar()
        num_batches += 1
        acc.update(label, out)
    return loss / num_batches, 1.0 - acc.get()[1]