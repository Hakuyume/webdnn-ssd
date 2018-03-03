import argparse
import numpy as np
import os

import chainer
import chainer.functions as F
from chainer import initializers
from chainer import variable
import chainercv
from webdnn.backend import generate_descriptor
from webdnn.frontend.chainer import ChainerConverter


class Normalize(chainer.Link):
    """chainercv.links.model.ssd.Normalize for WebDNN"""

    def __init__(self, n_channel, initial=0, eps=1e-5):
        super().__init__()
        self.eps = eps
        with self.init_scope():
            initializer = initializers._get_initializer(initial)
            self.scale = variable.Parameter(initializer)
            self.scale.initialize((n_channel),)

    def __call__(self, x):
        # replace chainer.functions.normalize with more basic operations
        norm = F.broadcast_to(
            F.sum(x * x, axis=1, keepdims=True) ** 0.5 + self.eps, x.shape)
        # insert a new axis first
        scale = F.broadcast_to(self.scale[None, :, None, None], x.shape)
        return x / norm * scale


class MultiboxDecoder(chainer.Link):
    """chainercv.links.model.ssd.MultiboxCoder for WebDNN"""

    def __init__(self, default_bbox, variance):
        super().__init__()
        with self.init_scope():
            self.default_bbox = variable.Parameter(default_bbox)
        self._variance = variance

    def __call__(self, mb_locs, mb_confs):
        # The following codes do not work well.
        # The coordinates of bounding boxes becomes very large.
        # yx = self.default_bbox[None, :, :2] + \
        #     mb_locs[:, :, :2] * self._variance[0] \
        #     * self.default_bbox[None, :, 2:]
        # hw = self.default_bbox[None, :, 2:] * \
        #     F.exp(mb_locs[:, :, 2:] * self._variance[1])

        yx = self.default_bbox[None, :, :2]
        hw = self.default_bbox[None, :, 2:]

        tl = yx - hw / 2
        br = yx + hw / 2

        mb_bboxes = F.concat((tl, br), axis=2)
        mb_scores = F.softmax(mb_confs, axis=2)[:, :, 1:]
        return mb_bboxes, mb_scores


class SSD300(chainer.Link):
    """chainercv.links.SSD300 for WebDNN"""

    def __init__(self):
        super().__init__()
        with self.init_scope():
            self.model = chainercv.links.SSD300(pretrained_model='voc0712')
            # copy parameters from MultiboxCoder
            self.decoder = MultiboxDecoder(
                self.model.coder._default_bbox.astype(np.float32),
                self.model.coder._variance)

        self.insize = self.model.insize

        # replace norm4 for WebDNN
        norm4_orig = self.model.extractor.norm4
        norm4 = Normalize(norm4_orig.scale.shape[0])
        norm4.eps = norm4_orig.eps
        norm4.copyparams(norm4_orig)
        self.model.extractor.norm4 = norm4

    def __call__(self, x):
        mb_bboxes, mb_scores = self.decoder(*self.model(x))
        print('{} -> {}, {}'.format(x.shape, mb_bboxes.shape, mb_scores.shape))
        return mb_bboxes, mb_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend',
                        choices=('webassembly'),
                        default='webassembly')
    parser.add_argument('--eigen')
    parser.add_argument('--out', default='model')
    args = parser.parse_args()

    model = SSD300()
    x = chainer.Variable(
        np.empty((1, 3, model.insize, model.insize), dtype=np.float32))
    mb_bboxes, mb_scores = model(x)

    graph = ChainerConverter().convert([x], [mb_bboxes, mb_scores])

    if args.eigen:
        os.environ['CPLUS_INCLUDE_PATH'] = args.eigen

    desc = generate_descriptor(args.backend, graph)
    desc.save(args.out)


if __name__ == '__main__':
    main()
