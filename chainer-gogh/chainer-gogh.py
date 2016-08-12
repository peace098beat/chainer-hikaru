#! coding:utf-8
import argparse

import numpy as np
from PIL import Image
import sys

from models import *


def subtract_mean(x0):
    x = x0.copy()
    x[0, 0, :, :] -= 120
    x[0, 1, :, :] -= 120
    x[0, 2, :, :] -= 120
    return x


def add_mean(x0):
    x = x0.copy()
    x[0, 0, :, :] += 120
    x[0, 1, :, :] += 120
    x[0, 2, :, :] += 120
    return x


def image_resize(img_file, width):
    gogh = Image.open(img_file)
    orig_w, orig_h = gogh.size[0], gogh.size[1]
    if orig_w > orig_h:
        new_w = width
        new_h = width * orig_h / orig_w
        gogh = np.asarray(gogh.resize((new_w, new_h)))[:, :, :3].transpose(
            2, 0, 1)[::-1].astype(np.float32)
        gogh = gogh.reshape((1, 3, new_h, new_w))
        print("image resized to: ", gogh.shape)
        hoge = np.zeros((1, 3, width, width), dtype=np.float32)
        hoge[0, :, width - new_h:, :] = gogh[0, :, :, :]
        gogh = subtract_mean(hoge)
    else:
        new_w = width * orig_w / orig_h
        new_h = width
        gogh = np.asarray(gogh.resize((new_w, new_h)))[:, :, :3].transpose(
            2, 0, 1)[::-1].astype(np.float32)
        gogh = gogh.reshape((1, 3, new_h, new_w))
        print("image resized to: ", gogh.shape)
        hoge = np.zeros((1, 3, width, width), dtype=np.float32)
        hoge[0, :, :, width - new_w:] = gogh[0, :, :, :]
        gogh = subtract_mean(hoge)
    return xp.asarray(gogh), new_w, new_h


def save_image(img, width, new_w, new_h, it):
    def to_img(x):
        im = np.zeros((new_h, new_w, 3))
        im[:, :, 0] = x[2, :, :]
        im[:, :, 1] = x[1, :, :]
        im[:, :, 2] = x[0, :, :]

        def clip(a):
            return 0 if a < 0 else (255 if a > 255 else a)

        im = np.vectorize(clip)(im).astype(np.uint8)
        Image.fromarray(im).save(args.out_dir + "/im_%05d.png" % it)

    if args.gpu >= 0:
        img_cpu = add_mean(img.get())
    else:
        img_cpu = add_mean(img)
    if width == new_w:
        to_img(img_cpu[0, :, width - new_h:, :])
    else:
        to_img(img_cpu[0, :, :, width - new_w:])


def get_matrix(y):
    ch = y.data.shape[1]
    wd = y.data.shape[2]
    gogh_y = F.reshape(y, (ch, wd ** 2))
    gogh_matrix = F.matmul(
        gogh_y, gogh_y, transb=True) / np.float32(ch * wd ** 2)
    return gogh_matrix


class Clip(chainer.Function):

    def forward(self, x):
        x = x[0]
        ret = cuda.elementwise(
            'T x', 'T ret',
            '''
                ret = x<-120?-120:(x>136?136:x);
            ''', 'clip')(x)
        return ret


def generate_image(img_orig, img_style, width, nw, nh, max_iter, lr, img_gen=None):
    mid_orig = nn.forward(Variable(img_orig, volatile=True))
    style_mats = [get_matrix(y)
                  for y in nn.forward(Variable(img_style, volatile=True))]

    if img_gen is None:
        if args.gpu >= 0:
            img_gen = xp.random.uniform(-20, 20,
                                        (1, 3, width, width), dtype=np.float32)
        else:
            img_gen = np.random.uniform(-20, 20,
                                        (1, 3, width, width)).astype(np.float32)
    img_gen = chainer.links.Parameter(img_gen)
    optimizer = optimizers.Adam(alpha=lr)
    optimizer.setup(img_gen)
    for i in range(max_iter):
        img_gen.zerograds()

        x = img_gen.W
        y = nn.forward(x)

        L = Variable(xp.zeros((), dtype=np.float32))
        for l in range(len(y)):
            ch = y[l].data.shape[1]
            wd = y[l].data.shape[2]
            gogh_y = F.reshape(y[l], (ch, wd ** 2))
            gogh_matrix = F.matmul(
                gogh_y, gogh_y, transb=True) / np.float32(ch * wd ** 2)

            L1 = np.float32(
                args.lam) * np.float32(nn.alpha[l]) * F.mean_squared_error(y[l], Variable(mid_orig[l].data))
            L2 = np.float32(nn.beta[l]) * F.mean_squared_error(gogh_matrix, Variable(style_mats[l].data)) / np.float32(
                len(y))
            L += L1 + L2

            if i % 100 == 0:
                print i, l, L1.data, L2.data

        L.backward()
        img_gen.W.grad = x.grad
        optimizer.update()

        tmp_shape = x.data.shape
        if args.gpu >= 0:
            img_gen.W.data += Clip().forward(img_gen.W.data).reshape(tmp_shape) - \
                img_gen.W.data
        else:
            def clip(x):
                return -120 if x < -120 else (136 if x > 136 else x)

            img_gen.W.data += np.vectorize(clip)(
                img_gen.W.data).reshape(tmp_shape) - img_gen.W.data

        if i % 100 == 0:
            save_image(img_gen.W.data, W, nw, nh, i)


import time


def proctimer(context):
    start_time = time.clock()

    def proctime():
        print(context + " %.3f" % (time.clock() - start_time) + "sec")
        return (time.clock() - start_time)

    return proctime

# ********************************************************
# Main
# ********************************************************
if __name__ == '__main__':
    import datetime

    ptimer_chainer_gogh = proctimer("chainer-gogh.py")

    parser = argparse.ArgumentParser(
        description='A Neural Algorithm of Artistic Style')
    parser.add_argument('--model', '-m', default='nin',
                        help='model file (nin, vgg, i2v, googlenet)')
    parser.add_argument('--orig_img', '-i', default='orig.png',
                        help='Original image')
    parser.add_argument('--style_img', '-s', default='style.png',
                        help='Style image')
    parser.add_argument('--out_dir', '-o', default='output',
                        help='Output directory')
    parser.add_argument('--gpu', '-g', default=-1, type=int,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--iter', default=5000, type=int,
                        help='number of iteration')
    parser.add_argument('--lr', default=4.0, type=float,
                        help='learning rate')
    parser.add_argument('--lam', default=0.005, type=float,
                        help='original image weight / style weight ratio')
    parser.add_argument('--width', '-w', default=256, type=int,
                        help='image width, height')
    parser.add_argument('--id', default=0, type=int, help="Exp Id")

    args = parser.parse_args()

    # Expname
    # ------------------------------------------------------------
    exp_name = ""
    exp_name += "%04d" % args.id
    exp_name += "_("
    exp_name += "-m_" + args.model
    exp_name += "-g_" + str(args.gpu)
    exp_name += "-iter_" + str(args.iter)
    exp_name += "-lr_" + str(args.lr)
    exp_name += "-lam_" + str(args.lam)
    exp_name += "-w_" + str(args.width)
    exp_name += ")"

    print exp_name

    # output dir
    # ------------------------------------------------------------
    # /output/
    out_root_dir = args.out_dir
    try:
        os.mkdir(args.out_dir)
    except:
        pass
    if os.path.exists(out_root_dir):
        with open(out_root_dir, 'ab') as fro:
            fro.write(datetime.datetime.today().strftime(
                "Analys Date Time : %Y-%m-%d %H:%M:%S") + "¥n")
            fro.write(exp_name + "¥n")

    args.out_dir = os.path.join(args.out_dir, exp_name)
    try:
        os.mkdir(args.out_dir)
    except:
        pass

    # LOGGER
    # ------------------------------------------------------------
    ERROR_LOG = "%s/log.txt" % (args.out_dir)

    def errorlog(e="", reset=False):
        if reset:
            with open(ERROR_LOG, 'wb') as fe:
                fe.write(e + "\n")
        else:
            with open(ERROR_LOG, 'ab') as fe:
                fe.write(e + "\n")

    # PARAMETER LOG
    # ------------------------------------------------------------
    HIKARU_AI_LOGO = """
***********************************************************
 *   * ***** *  **  ****   ****   *    **     *   *****
 *   *   *   *  *   *   *  *   *  *    *     * *    *
 *****   *   ***    *    * *****  *    *    *   *   *
 *   *   *   * *    ***  * *   *  *    *    *****   *
 *   * ***** *  **  *    * *   **  ****     *   * ***** *
***********************************************************
"""
    errorlog(HIKARU_AI_LOGO, True)
    errorlog(datetime.datetime.today().strftime(
        "Analys Date Time : %Y-%m-%d %H:%M:%S"))
    errorlog()
    errorlog("id : %03d" % args.id)
    errorlog("model : %s" % args.model)
    errorlog("orig_img : " + args.orig_img)
    errorlog("style_img : " + args.style_img)
    errorlog("out_dir : " + args.out_dir)
    errorlog("gpu : %d" % args.gpu)
    errorlog("iter : %d" % args.iter)
    errorlog("lr : %f" % args.lr)
    errorlog("lam : %f" % args.lam)
    errorlog("width : %d" % args.width)
    errorlog()

    # TEST FILE EXISTS
    # ------------------------------------------------------------
    # ORING IMAGE EXISTS TEST
    if not os.path.exists(args.orig_img):
        errorlog(">> ERROR ... orig_img not found : %s" % (args.orig_img))
        errorlog(">> ERROR ... orig_img not found : %s" %
                 (os.path.realpath(args.orig_img)))
        sys.exit(201)
    # STYLE IMAGE EXISTS TEST
    if not os.path.exists(args.style_img):
        errorlog(">> ERROR ... style not found : %s" % (args.style_img))
        errorlog(">> ERROR ... style not found : %s" %
                 (os.path.realpath(args.style_img)))
        sys.exit(202)

    # CUDA Setting
    # ------------------------------------------------------------
    if args.gpu >= 0:
        cuda.check_cuda_available()
        chainer.Function.type_check_enable = False
        cuda.get_device(args.gpu).use()
        xp = cuda.cupy
    else:
        xp = np

    # MODEL LOAD
    # ------------------------------------------------------------
    # ROOT DIR
    root_dir = os.path.dirname(os.path.abspath(__file__))

    if 'nin' in args.model:
        fn = os.path.join(root_dir, "nin_imagenet.caffemodel")
    elif 'vgg' in args.model:
        fn = os.path.join(root_dir, "VGG_ILSVRC_16_layers.caffemodel")
    elif 'i2v' in args.model:
        fn = os.path.join(root_dir, "illust2vec_tag_ver200.caffemodel")
    elif 'googlenet' in args.model:
        fn = os.path.join(root_dir, "bvlc_googlenet.caffemodel")
    else:
        fn = None

    if not os.path.exists(fn):
        errorlog(">> ERROR ... model not found : %s" % (fn))
        fn = None

    if 'nin' in args.model:
        nn = NIN(fn)
    elif 'vgg' in args.model:
        nn = VGG(fn)
    elif 'i2v' in args.model:
        nn = I2V(fn)
    elif 'googlenet' in args.model:
        nn = GoogLeNet(fn)
    else:
        print 'invalid model name. you can use (nin, vgg, i2v, googlenet)'
        errorlog('invalid model name. you can use (nin, vgg, i2v, googlenet)')
    if args.gpu >= 0:
        nn.model.to_gpu()

    ptimer_chainer_gogh()

    # Resize
    W = args.width
    img_content, nw, nh = image_resize(args.orig_img, W)
    img_style, _, _ = image_resize(args.style_img, W)

    generate_image(img_content, img_style, W, nw, nh,
                   img_gen=None, max_iter=args.iter, lr=args.lr)

    # Process End
    dt = ptimer_chainer_gogh()
    errorlog("> Iter Times %f[s]" % (dt,))
