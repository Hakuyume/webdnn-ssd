import argparse
import numpy as np
import os

import chainer
import chainer.functions as F
from chainer import variable
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
        mb_locs = []
        mb_confs = []

        for i, x in enumerate(xs):
            mb_loc = self.base.loc[i](x)
            mb_loc = F.transpose(mb_loc, (0, 2, 3, 1))
            mb_loc = F.reshape(mb_loc, (mb_loc.shape[0], -1, 4))
            mb_locs.append(mb_loc)

            mb_conf = self.base.conf[i](x)
            mb_conf = F.transpose(mb_conf, (0, 2, 3, 1))
            mb_conf = F.reshape(
                mb_conf, (mb_conf.shape[0], -1, self.base.n_class))
            mb_confs.append(mb_conf)

        # prevent concat
        return mb_locs, mb_confs


class MultiboxDecoder(chainer.Link):
    """chainercv.links.model.ssd.MultiboxCoder for WebDNN"""

    def __init__(self, base):
        super().__init__()
        with self.init_scope():
            self.default_bbox = variable.Parameter(
                base._default_bbox.astype(np.float32))
        self._variance = base._variance

    def __call__(self, mb_locs, mb_confs):
        mb_bboxes = []
        mb_scores = []

        k = 0
        for mb_loc, mb_conf in zip(mb_locs, mb_confs):
            default_bbox = self.default_bbox[k:k + mb_loc.shape[1]]
            k += mb_loc.shape[1]

            yx = default_bbox[None, :, :2] + \
                mb_loc[:, :, :2] * self._variance[0] \
                * default_bbox[None, :, 2:]
            hw = default_bbox[None, :, 2:] * \
                F.exp(mb_loc[:, :, 2:] * self._variance[1])

            tl = yx - hw / 2
            br = yx + hw / 2

            mb_bboxes.append(F.concat((tl, br), axis=2))
            mb_scores.append(F.softmax(mb_conf, axis=2)[:, :, 1:])

        return mb_bboxes, mb_scores


class SSD300(chainer.Link):
    """chainercv.links.SSD300 for WebDNN"""

    def __init__(self, base):
        super().__init__()
        with self.init_scope():
            self.base = base
            self.multibox = Multibox(self.base.multibox)
            self.decoder = MultiboxDecoder(self.base.coder)

        self.insize = self.base.insize

        # replace norm4 for WebDNN
        norm4_orig = self.base.extractor.norm4
        norm4 = Normalize(norm4_orig.scale.shape[0])
        norm4.eps = norm4_orig.eps
        norm4.copyparams(norm4_orig)
        self.base.extractor.norm4 = norm4

    def __call__(self, x):
        mb_bboxes, mb_scores = self.decoder(
            *self.multibox(self.base.extractor(x)))
        return mb_bboxes, mb_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',
                        choices=('webgl', 'webassembly'),
                        nargs='*', default=())
    parser.add_argument('--eigen')
    parser.add_argument('--out', default='model')
    args = parser.parse_args()

    model = SSD300(chainercv.links.SSD300(pretrained_model='voc0712'))
    x = chainer.Variable(
        np.empty((1, 3, model.insize, model.insize), dtype=np.float32))
    mb_bboxes, mb_scores = model(x)
    ys = sum(zip(mb_bboxes, mb_scores), ())

    print(x.shape, '->', ', '.join('{}'.format(y.shape) for y in ys))

    graph = ChainerConverter().convert([x], ys)

    if args.eigen:
        os.environ['CPLUS_INCLUDE_PATH'] = args.eigen
    for backend in args.backend:
        print('backend:', backend)
        desc = generate_descriptor(backend, graph)
        desc.save(args.out)


if __name__ == '__main__':
    main()
