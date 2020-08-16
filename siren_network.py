import numpy as np
import mxnet as mx
from mxnet import nd, gluon
from mxnet.gluon import nn


class SirenInit(mx.init.Initializer):
    def _init_weight(self, name, data):
        num_input = data.shape[-1]
        data[:] = nd.random.uniform(low=-np.sqrt(6 / num_input), high=np.sqrt(6 / num_input), shape=data.shape)


class SirenNetwork(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(SirenNetwork, self).__init__(**kwargs)

        with self.name_scope():
            self.fc_1 = nn.Dense(256, weight_initializer=SirenInit())
            self.fc_2 = nn.Dense(128, weight_initializer=SirenInit())
            self.fc_3 = nn.Dense(128, weight_initializer=SirenInit())
            self.fc_4 = nn.Dense(64, weight_initializer=SirenInit())
            self.fc_5 = nn.Dense(32, weight_initializer=SirenInit())
            self.fc_6 = nn.Dense(3, weight_initializer=SirenInit())

    def hybrid_forward(self, F, x, *args, **kwargs):
        x = self.fc_1(x)
        x = F.sin(30.0 * x)
        x = self.fc_2(x)
        x = F.sin(1.0 * x)
        x = self.fc_3(x)
        x = F.sin(1.0 * x)
        x = self.fc_4(x)
        x = F.sin(1.0 * x)
        x = self.fc_5(x)
        x = F.sin(1.0 * x)
        x = self.fc_6(x)
        x = F.sin(1.0 * x)
        return x