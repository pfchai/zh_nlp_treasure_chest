# -*- coding: utf-8 -*-

import os


def get_project_path():
    current_path = os.path.abspath(__file__)
    current_dir = os.path.split(current_path)[0]
    return os.path.abspath(os.path.join(current_dir, '../..'))

