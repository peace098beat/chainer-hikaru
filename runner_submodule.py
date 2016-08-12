#! coding:utf-8
'''
runner_submodule.py

runner.pyのサブ関数

'''
# print(__file__)
import ConfigParser
import json
import os

import time


def proctimer(context):
    start_time = time.clock()

    def proctime():
        print(context + " %.3f" % (time.clock() - start_time) + "sec")

    return proctime


def cartesian_product(X):
    """直積を求める
    直積とは A={a, b, c}, B={d, e}のような2つの集合が与えられたとすると
    A×B={(a,d),(a,e),(b,d),(b,e),(c,d),(c,e)}
    """
    import itertools

    Y = list(itertools.product(*X))
    # >> [(11, 21, 31), (11, 21, 32), (11, 21, 33), (11, 22, 31), (11, 22, 32), (11, 22, 33),
    return Y


def test_cartesian_product():
    X = [
        [11, 12, 13],
        [21, 22, 23],
        [31, 32, 33],
    ]
    x_size = [3, 3, 3]
    x_length = 3 * 3 * 3
    Y = [
        (11, 21, 31),
        (11, 21, 32),
        (11, 21, 33),
        (11, 22, 31),
        (11, 22, 32),
        (11, 22, 33),
        (11, 23, 31),
        (11, 23, 32),
        (11, 23, 33),
        (12, 21, 31),
        (12, 21, 32),
        (12, 21, 33),
        (12, 22, 31),
        (12, 22, 32),
        (12, 22, 33),
        (12, 23, 31),
        (12, 23, 32),
        (12, 23, 33),
        (13, 21, 31),
        (13, 21, 32),
        (13, 21, 33),
        (13, 22, 31),
        (13, 22, 32),
        (13, 22, 33),
        (13, 23, 31),
        (13, 23, 32),
        (13, 23, 33),
    ]
    y_length = len(Y)
    assert y_length == x_length

    ans = cartesian_product(X)
    assert Y == ans, (ans, Y)
    assert len(Y) == len(ans), (len(Y), len(ans))

    def matched_list(Y1, Y2):
        result = []
        for y1 in Y1:
            for y2 in Y2:
                if y1 == y2:
                    assert y1 == y2, (y1, y2)
                    result.append(y1)
        return result

    Y_matched = matched_list(Y, ans)
    assert len(Y) == len(Y_matched), (y_length, len(Y_matched))


test_cartesian_product()


def calc_subprocess(cmd):
    import subprocess

    try:
        subprocess.check_call(cmd)
        return_code = 1
    except subprocess.CalledProcessError as e:
        print(">>CalledProcessError!! returncode:%r, args:%r, cmd:%r, msg:%r, output:%r" % (
            e.returncode, e.args, e.cmd, e.message, e.output))
        return_code = -1
    except OSError as e:
        print(">>ERROR!! return errno:%r, args:%r, filename:%r, msg:%r, strerror:%r" % (
            e.errno, e.args, e.filename, e.message, e.strerror))
        return_code = -1

    return return_code


def calc_subprocess2(cmd):
    import subprocess

    print("calc_subprocess")
    try:
        proc = subprocess.Popen([cmd],
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                )
        sout, serr = proc.communicate()
        print("sout :" ,sout)
    except subprocess.CalledProcessError as e:
        print(u">>CalledProcessError!! returncode:%r, args:%r, cmd:%r, msg:%r, output:%r" % (
            e.returncode, e.args, e.cmd, e.message, e.output))
    except OSError as e:
        print(u">>OS ERROR!! return errno:%r, args:%s, filename:%r, msg:%r, strerror:%r" % (
            e.errno, e.args, e.filename, e.message, e.strerror))
        print("%s" % e.strerror)
        with open('error.log', 'ab') as fe:
            fe.write(e.strerror)

    return 1


def config2dict(config_path):
    assert os.path.exists(config_path)
    if not os.path.exists(config_path):
        raise IOError("not found config %s"%(config_path,))

    # Parse Config
    config = ConfigParser.RawConfigParser()
    config.read(config_path)

    config_dict = dict()
    for section in config.sections():
        config_dict[section] = dict()
        for option in config.options(section):
            value = json.loads(config.get(section, option))
            config_dict[section][option] = value

    return config_dict


def dict_savecsv(dictionary, csv_path):
    fp = open(csv_path, 'wb')

    for section in dictionary.keys():
        fp.write("\n")
        fp.write(section)
        fp.write("\n")

        for label, value in dictionary[section].items():
            if not isinstance(value, list):
                value = list((value,))
            s = label + ' : ' + ", ".join(map(str, value))
            s += "\n"
            fp.write(s)


def convert_config(config_path):
    assert os.path.exists(config_path)
    config_dict = config2dict(config_path)
    dict_savecsv(config_dict, "./config.log")

    param_dict = config_dict["parameter"]
    param_keys = param_dict.keys()
    param_values = param_dict.values()
    print param_keys
    print param_values

    params = cartesian_product(param_values)

    input_path = "parameter.log"
    fp = open(input_path, 'wb')

    fp.write(str(param_keys))
    fp.write("\n")

    header = "Params Number :%d" % (len(params),)
    fp.write(header)
    fp.write("\n")

    for p in params:
        fp.write(str(p))
        fp.write("\n")



if __name__ == '__main__':
    convert_config("config.cfg")
