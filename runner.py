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
import datetime
LOG_RUNNER = "./runner.log"


def now_date():
    return datetime.datetime.today().strftime("Date Time : %Y-%m-%d %H:%M:%S")


def start_process():
    ret = 'Starting : %s'%(multiprocessing.current_process().name,)
    runner_log(ret)
    print(ret)


def runner_log(s, clean=False):
    if not isinstance(s, str):
        return

    if clean:
        with open(LOG_RUNNER, 'wb') as fl:
            fl.write(s + "\n")
            return 0

    with open(LOG_RUNNER, 'ab') as fl:
        fl.write(s + "\n")

if __name__ == '__main__':

    runner_log("", True)
    runner_log(now_date())

    # PYTHON 64BIT
    is_64bits = sys.maxsize > 2 ** 32
    if not is_64bits:
        print("Python 32bit cant work vgg")
        runner_log("Python isnot 64bit... 32bit cant work vgg")
        sys.exit(32)
    else:
        runner_log("Python is 64bit")

    # OS Check
    if submod.is_windows():
        runner_log("OS is Windows")
    else:
        runner_log("OS is not Windows : %s" % (os.uname(),))

    # PARSE COMMAND LINE
    parser = argparse.ArgumentParser(
        description='A Neural Algorithm of Artistic Style')
    parser.add_argument(
        '--cfg', '-c', default="config.cfg", help="config file")
    args = parser.parse_args()

    # LOG DIR
    try:
        os.mkdir("./log")
    except:
        pass
    runner_log("> PARSE COOMAND LINE")

    # CREATE COMMAND
    # -----------------------------------------------------------------------------
    # コンフィグの呼び出し
    config_path = args.cfg
    config_dict = submod.config2dict(config_path)
    submod.dict_savecsv(config_dict, "./log/config.log")
    runner_log("> config loaded ... : %s" % (config_path,))
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
    # -----------------------------------------------------------------------------
    with open("./log/command.log", "wb") as fc:
        fc.write("\n".join(commands))

    # RUN PROCESSES
    # -----------------------------------------------------------------------------
    runner_log("> PROCESS START")
    # プロセスサイズ
    pool_size = config_dict["common"]["pool_size"]
    # プロセスプール
    pool = multiprocessing.Pool(processes=pool_size, initializer=start_process)
    # 実際の処理
    pool_outputs = pool.map(submod.calc_subprocess, commands)
    print 'Pool    :', pool_outputs
    # おまじない
    pool.close()  # no more tasks
    pool.join()  # wrap up current tasks


    # try:
    #     # 実際の処理
    #     pool_outputs = pool.map(submod.calc_subprocess, commands)
    #     print 'Pool    :', pool_outputs
    # except KeyboardInterrupt:
    #     # おまじない
    #     pool.close()  # no more tasks
    #     # pool.join()  # wrap up current tasks
