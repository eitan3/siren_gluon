import numpy as np
import mxnet as mx
from mxnet import nd, autograd, gluon
from siren_network import SirenNetwork
from data_loader import DataIter
from utils import evaluate_accuracy
import time
import configargparse

p = configargparse.ArgumentParser()
p.add('-i', '--image_path', required=True, help='Image file to load.')
args = p.parse_args()

np.random.seed(1)
mx.random.seed(1)
ctx = mx.gpu(0)

batch_size = 32
epochs = 300

lr = 0.0001
wd = 0.0

train_iter = DataIter(ctx, batch_size, args.image_path)

net = SirenNetwork()
loss_function = gluon.loss.L2Loss()

net.collect_params().initialize(mx.init.Xavier(), ctx=ctx)

trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': lr,
                                                       'wd': wd})

smoothing_constant = .01
moving_loss = 0
start_time = time.time()

for i in range(epochs):
    print("=========================================")
    print("Epoch: %s" % i)
    for j, (data, label) in enumerate(train_iter):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = loss_function(output, label)
        loss.backward()
        trainer.step(data.shape[0])

        curr_loss = nd.mean(loss).asscalar()
        moving_loss = (curr_loss if ((i == 0) and (j == 0))
                       else (1 - smoothing_constant) * moving_loss + smoothing_constant * curr_loss)

        if j % 100 == 0:
            print("Iter %i/%i, Loss: %.6f" % (j, train_iter.num_batches,curr_loss))

    train_cumulative_loss, train_acc = evaluate_accuracy(train_iter, net, loss_function, ctx)
    # val_cumulative_loss, val_acc, val_auc = evaluate_accuracy(val_iter, net, ctx)

    print("Epoch: %s, Loss: %.6f" % (i, moving_loss))
    print("Train_loss %.6f, Train_acc: %.8f" % (train_cumulative_loss, train_acc))
    # print("Val_loss %.6f, Val_acc: %.3f, Val_auc: %.3f" % (val_cumulative_loss, val_acc, val_auc))
    net.save_parameters("./checkpoints/model_"+str(i+1)+".params")

print("--- %s seconds ---" % (time.time() - start_time))
