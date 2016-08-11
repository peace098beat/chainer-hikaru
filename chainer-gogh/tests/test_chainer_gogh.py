# 1 coding:utf-8
"""

python -m unittest discover -s tests
"""
import sys
import os
import unittest

sys.path.append("..")


class TestChainerGogh(unittest.TestCase):
    def setUp(self):
        pass

    def test_load_model(self):
        os.chdir(os.path.join(os.path.dirname(__file__), '..'))
        model_names = ("nin_imagenet.caffemodel",
                       "VGG_ILSVRC_16_layers.caffemodel",
                       "illust2vec_tag_ver200.caffemodel",
                       "bvlc_googlenet.caffemodel",
                       "fifi.caffemodel")
        root_dir = os.path.join(os.path.dirname(__file__), '..')
        model_paths = [os.path.join(root_dir, m) for m in model_names]
        self.assertTrue(os.path.exists(model_paths[0]))
        self.assertTrue(os.path.exists(model_paths[1]))
        self.assertTrue(os.path.exists(model_paths[2]))
        self.assertTrue(os.path.exists(model_paths[3]))

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
