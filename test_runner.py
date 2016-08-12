#! coding:utf-8
"""
test_runner.py

Created by 0160929 on 2016/08/11 14:36
"""

import sys
import os
import unittest

from runner_submodule import calc_subprocess


flag_win32 = False

try:
    os.uname()
except AttributeError:
    flag_win32 = True

if flag_win32:
    # (Windowsの場合の処理）
    pass
else:
    # (Linuxの場合の処理）
    pass


class TestRunner(unittest.TestCase):
    def setUp(self):
        pass

    def test_subprocess(self):
        cmd = "ls -al"
        return_code = calc_subprocess(cmd)
        self.assertEqual(return_code, 1)

    def test_cpu_cfg(self):
        _cmd = "python run_runner.py -c test_cpu.cfg"
        return_code = calc_subprocess(_cmd)
        self.assertEqual(return_code, 1)

    def test_gpu_cfg(self):
        _cmd = "python run_runner.py -c test_gpu.cfg"
        return_code = calc_subprocess(_cmd)
        self.assertEqual(return_code, 1)

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
