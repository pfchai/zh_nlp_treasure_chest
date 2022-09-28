# -*- coding: utf-8 -*-
import os
import multiprocessing


def set_gpu_memory_growth():
    "设置"
    
    import tensorflow as tf

    tf_versions = tf.__version__.split('.')
    if int(tf_versions[0]) == 2:
        if int(tf_versions[1]) in (0, 1):
            tf.config.gpu.set_per_process_memory_growth(True)
        else:
            gpus = tf.config.experimental.list_physical_devices('GPU')
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)


def force_use_cpu():
    "强制不使用GPU，需要在代码最前面使用"
    os.environ['CUDA_VISIBLE_DEVICES'] = ''






def run_in_new_process(func):
    """
    在单独的进程中执行函数
    目的：模型训练完成后，不继续占用GPU
    来源：https://github.com/tensorflow/tensorflow/issues/36465#issuecomment-582749350
    """
    def wrapper(*args, **kw):
        process_eval = multiprocessing.Process(target=func, args=args, kwargs=kw)
        process_eval.start()
        process_eval.join()
    return wrapper


