# -*- coding: utf-8 -*-

import os

from keras.callbacks import ModelCheckpoint, TensorBoard


class Trainer():
    """
    主要是初始化一些目录
    """
    def __init__(self, model_name='base', dataset=None, tokenizer=None, model=None):
        self.model_name = model_name

        # 相关文件路径设置
        self.base_output_dir = './outputs'
        self.checkpoint_dir = os.path.join(self.base_output_dir, 'checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.log_dir = os.path.join(self.base_output_dir, 'logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # 组件
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.model = model

    def train(self, fit_args):
        train_x, train_y, valid_x, valid_y = self.dataset.get_train_input_data(self.tokenizer)

        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f'{self.model_name}_' + '{epoch:03d}_acc{val_accuracy:.2f}.hdf5'
        )
        checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_accuracy', verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=self.log_dir)

        self.model.fit(
            train_x, train_y,
            validation_data=(valid_x, valid_y),
            callbacks=[checkpoint, tensorboard],
            **fit_args
        )
