rem working test
rem python chainer-gogh.py -m nin -i ./sample_images/cat.png -s ./sample_gibri/style_3.png -o output_dir_gibri2 -g -1
rem python chainer-gogh.py -m nin -i ./sample_images/cat.png -s ./sample_images/style_0.png -o output_dir -g -1
rem python chainer-gogh.py -m nin -i ./sample_images/cat.png -s ./sample_gibri/style_hikaru_1.jpg -o output_dir_hikaru0 -g -1
rem python chainer-gogh.py -m nin -i ./sample_images/cat.png -s ./sample_gibri/style_hikaru_1s.jpg -o output_dir_hikaru1 -g -1
rem python chainer-gogh.py -m nin -i ./sample_hikaru/Hikaru-A003.jpg -s ./sample_hikaru/Hikaru-A001.jpg -o output_dir_hikaru_2 -g -1
rem python chainer-gogh.py -m nin -i ./sample_hikaru/Hikaru-A002.jpg -s ./sample_hikaru/Hikaru-A001.jpg -o output_dir_hikaru_3 -g -1
rem python chainer-gogh.py -m nin -i ./sample_hikaru/Hikaru-A002.jpg -s ./sample_hikaru/Hikaru-A001.jpg -o output_dir_hikaru_3 -g -1

rem Test NN Model Subset
rem python chainer-gogh.py -m nin -i ./sample_images/cat.png -s ./sample_images/style_0.png -o output_nin_s0
rem python chainer-gogh.py -m googlenet -i ./sample_images/cat.png -s ./sample_images/style_0.png -o output_ggn_s0
rem python chainer-gogh.py -m vgg -i ./sample_images/cat.png -s ./sample_images/style_0.png -o output_vgg_s0
rem python chainer-gogh.py -m i2v -i ./sample_images/cat.png -s ./sample_images/style_0.png -o output_i2v_s0
