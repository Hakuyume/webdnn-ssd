import argparse
import numpy as np
import os

import chainer
import chainer.functions as F
from chainer import initializers
from chainer import variable
from chainercv.links import SSD300
from webdnn.backend import generate_descriptor
from webdnn.frontend.chainer import ChainerConverter


class Normalize(chainer.Link):
    """chainercv.ssd.links.model.ssd.Normalize for WebDNN"""

    def __init__(self, n_channel, initial=0, eps=1e-5):
        super(Normalize, self).__init__()
        self.eps = eps
        with self.init_scope():
            initializer = initializers._get_initializer(initial)
            self.scale = variable.Parameter(initializer)
            self.scale.initialize((n_channel),)

    def __call__(self, x):
        # replace chainer.functions.normalize with more basic operations
        norm = F.broadcast_to(
            F.sum(x * x, axis=1, keepdims=True) ** 0.5 + self.eps, x.shape)
        # insert np.newaxis first
        scale = F.broadcast_to(
            self.scale[np.newaxis, :, np.newaxis, np.newaxis], x.shape)
        return x / norm * scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',
                        choices=('webassembly'),
                        default='webassembly')
    parser.add_argument('--eigen')
    parser.add_argument('--out', default='ssd300')
    args = parser.parse_args()

    model = SSD300(pretrained_model='voc0712')
    # replace norm4 for WebDNN
    norm4_orig = model.extractor.norm4
    norm4 = Normalize(norm4_orig.scale.shape[0])
    norm4.eps = norm4_orig.eps
    norm4.copyparams(norm4_orig)
    model.extractor.norm4 = norm4

    x = chainer.Variable(
        np.empty((1, 3, model.insize, model.insize), dtype=np.float32))
    mb_locs, mb_confs = model(x)

    graph = ChainerConverter().convert([x], [mb_locs, mb_confs])

    if args.eigen:
        os.environ['CPLUS_INCLUDE_PATH'] = args.eigen

    desc = generate_descriptor(args.backend, graph)
    desc.save(args.out)


if __name__ == '__main__':
    main()
