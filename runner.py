#! coding:utf-8
"""
runner.py

Created by 0160929 on 2016/08/10 17:05
"""
import argparse
import multiprocessing
import os
import sys

import runner_submodule as submod


def start_process():
    print 'Starting', multiprocessing.current_process().name


LOG_RUNNER = "./log/runner.log"

def runner_log(s, clean=False):
    if clean:
        with open(LOG_RUNNER, 'wb') as fl:
            fl.write(s + "\n")
            return 0

    with open(LOG_RUNNER, 'ab') as fl:
        fl.write(s + "\n")

if __name__ == '__main__':



    runner_log("", True)


    # PYTHON 64BIT
    is_64bits = sys.maxsize > 2 ** 32
    if not is_64bits:
        print("Python 32bit cant work vgg")
        sys.exit(100)

    # PARSE COMMAND LINE
    parser = argparse.ArgumentParser(
        description='A Neural Algorithm of Artistic Style')
    parser.add_argument('--cfg', '-c', default="config.cfg", help="config file")
    args = parser.parse_args()

    # LOG DIR
    try:
        os.mkdir("./log")
    except:
        pass
    runner_log("PARSE COOMAND LINE")

    # CREATE COMMAND
    # -----------------------------------------------------------------------------
    # コンフィグの呼び出し
    config_path = args.cfg
    config_dict = submod.config2dict(config_path)
    submod.dict_savecsv(config_dict, "./log/config.log")
    # パラメータの取得
    param_dict = config_dict["parameter"]
    # 直積のリスト
    header = param_dict.keys()
    params = submod.cartesian_product(param_dict.values())
    assert len(header) == len(params[0])
    # コマンドの生成
    commands = []
    for no, param in enumerate(params):
        s = "python " + config_dict["common"]["pyfile"] + ""
        # param option
        for h, p in zip(header, param):
            s += " %s %s" % (h, p)  # option をつなげる
        s += " --id " + str(no)
        commands.append(s)

    # コマンドの保存
    with open("./log/command.log", "wb") as fc:
        fc.write("\n".join(commands))

    # -----------------------------------------------------------------------------
    runner_log(">> PROCESS START")
    # プロセスサイズ
    pool_size = config_dict["common"]["pool_size"]
    # プロセスプール
    pool = multiprocessing.Pool(processes=pool_size, initializer=start_process)
    # 実際の処理
    pool_outputs = pool.map(submod.calc_subprocess, commands)
    # おまじない
    pool.close()  # no more tasks
    pool.join()  # wrap up current tasks
    print 'Pool    :', pool_outputs
