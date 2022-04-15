# -*- coding: utf-8 -*-

import argparse

from zh_nlp_demo.keras.data.dataset.msra import MSRA


def create_parser(description='命名实体识别'):
    support_dataset = (
        'msra',
    )
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--dataset', type=str, default='msra',
        choices=support_dataset, help='指定数据集'
    )
    parser.add_argument('--do_train', action='store_true', default=False)
    parser.add_argument('--do_test', action='store_true', default=False)
    return parser


def create_dataset(args):
    if args.dataset == 'msra':
        dataset = MSRA()
    else:
        raise '不支持的数据集'
    return dataset
