import numpy as np
import mxnet as mx
from siren_network import SirenNetwork
from utils import create_x_input, norm_coord_to_real
import time
import cv2
import configargparse

p = configargparse.ArgumentParser()
p.add('-m', '--model_path', required=True, help='Model file to load.')
args = p.parse_args()

np.random.seed(1)
mx.random.seed(1)
ctx = mx.gpu(0)

# (Height, Width, Channels)
img_size = (286, 286, 3)
batch_size = 256

net = SirenNetwork()
net.load_parameters(args.model_path, ctx=ctx)

x_inputs = create_x_input(img_size)
out_image = np.zeros(img_size)

samples_counter = 0
start_time = time.time()
while samples_counter < x_inputs.shape[0]:
    bs = batch_size
    if samples_counter + batch_size >= x_inputs.shape[0]:
        bs = x_inputs.shape[0] - samples_counter

    x_in = x_inputs[samples_counter:samples_counter + bs, :]
    x_in_ctx = mx.nd.array(x_in, ctx=ctx)

    y_out = net(x_in_ctx).asnumpy()

    for j in range(bs):
        x_real = norm_coord_to_real(x_in[j], img_size)
        out_image[x_real[0], x_real[1], :] = y_out[j, :]

    samples_counter += bs

out_image = out_image * 255.0
cv2.imwrite("result.png", out_image)
print("--- %s seconds ---" % (time.time() - start_time))
