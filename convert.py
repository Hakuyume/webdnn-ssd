import argparse
import numpy as np

import chainer
import chainer.functions as F
import chainercv
from webdnn.backend import generate_descriptor
from webdnn.frontend.chainer import ChainerConverter


class Normalize(chainercv.links.model.ssd.Normalize):
    """chainercv.links.model.ssd.Normalize for WebDNN"""

    def __call__(self, x):
        # replace chainer.functions.normalize with more basic operations
        norm = F.broadcast_to(
            F.sum(x * x, axis=1, keepdims=True) ** 0.5 + self.eps, x.shape)
        # insert a new axis first
        scale = F.broadcast_to(self.scale[None, :, None, None], x.shape)
        return x / norm * scale


class Multibox(chainer.Link):
    """chainercv.links.model.ssd.Multibox for WebDNN"""

    def __init__(self, base):
        super().__init__()
        with self.init_scope():
            self.base = base

    def __call__(self, xs):
        ys = []
        for i, x in enumerate(xs):
            ys.append(self.base.loc[i](x))
            ys.append(self.base.conf[i](x))
        return ys


class SSD(chainer.Link):
    """chainercv.links.model.ssd.SSD for WebDNN"""

    def __init__(self, base):
        super().__init__()
        with self.init_scope():
            self.base = base
            self.multibox = Multibox(self.base.multibox)

        self.insize = self.base.insize

        if hasattr(self.base.extractor, 'norm4'):
            # replace norm4 for WebDNN
            norm4_orig = self.base.extractor.norm4
            norm4 = Normalize(norm4_orig.scale.shape[0])
            norm4.eps = norm4_orig.eps
            norm4.copyparams(norm4_orig)
            self.base.extractor.norm4 = norm4

    def __call__(self, x):
        return self.multibox(self.base.extractor(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',
                        choices=('webgl', 'webassembly'),
                        nargs='+')
    parser.add_argument('--out', default='model')
    args = parser.parse_args()

    model = SSD(chainercv.links.SSD300(pretrained_model='voc0712'))
    x = chainer.Variable(
        np.empty((1, 3, model.insize, model.insize), dtype=np.float32))
    ys = model(x)
    print(x.shape, '->', ', '.join('{}'.format(y.shape) for y in ys))

    graph = ChainerConverter().convert([x], ys)

    for backend in args.backend:
        print('backend:', backend)
        desc = generate_descriptor(backend, graph)
        desc.save(args.out)


if __name__ == '__main__':
    main()
