# -*- coding: utf-8 -*-

import os


def get_project_path():
    current_path = os.path.abspath(__file__)
    current_dir = os.path.split(current_path)[0]
    return os.path.abspath(os.path.join(current_dir, '..'))


def get_default_dict_path():
    project_path = get_project_path()
    return os.path.join(project_path, 'data', 'dict', 'vocab.txt')


def get_default_char_dict_path():
    project_path = get_project_path()
    return os.path.join(project_path, 'data', 'dict', 'char_vocab.txt')