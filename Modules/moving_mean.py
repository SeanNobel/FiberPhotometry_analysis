# -*- coding: utf-8 -*-
import numpy as np

class MovingMeanAlgorithm:
    def __init__(self, length, window):
        self.len = length
        self.window = window

    def __call__(self, raw_data):
        processed_data = np.empty(self.len)

        for i in range(self.len):
            if i < self.window:
                processed_data[i] = raw_data[i]
            else:
                x = 0
                for j in range(self.window):
                    x += raw_data[i - j]
                x /= self.window
                processed_data[i] = x

        return processed_data

# ウィンドウの長さよりも小さい最初のデータはそのままにしている
