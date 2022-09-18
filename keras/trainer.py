# -*- coding: utf-8 -*-

import os

from keras.models import load_model
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
        self.monitor = 'val_accuracy'

    def train(self, fit_args):
        train_x, train_y, valid_x, valid_y = self.dataset.get_train_input_data(self.tokenizer)

        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f'{self.model_name}_' + '{epoch:03d}_acc{' + self.monitor + ':.2f}.hdf5'
        )
        checkpoint = ModelCheckpoint(checkpoint_file, monitor=self.monitor, verbose=1, save_best_only=True)
        tensorboard = TensorBoard(log_dir=self.log_dir)

        self.model.fit(
            train_x, train_y,
            validation_data=(valid_x, valid_y),
            callbacks=[checkpoint, tensorboard],
            **fit_args
        )

    def test(self, checkpoint_name):
        # model = load_model(os.path.join(self.checkpoint_dir, 'best_model.hdf5'))
        self.model.load_weights(os.path.join(self.checkpoint_dir, checkpoint_name))
        test_x, test_y = self.dataset.get_test_input_data(self.tokenizer)

        pred_result = self.model.predict(test_x)
        pred_y = pred_result.argmax(axis=1)
        test_y = test_y.argmax(axis=1)
        print(classification_report(test_y, pred_y))
