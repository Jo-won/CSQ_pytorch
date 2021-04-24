import numpy as np
import cv2
class TSNSampling(object):
    def __init__(self, num, random_shift, seed=0):
        self.num = num
        self.random_shift = random_shift
        self.rng = np.random.RandomState(seed)
    def sampling(self, range_max, index=None, v_id=None):
        if index is not None:
            range_max = index.shape[0]
        if self.random_shift:
            average_duration = range_max // self.num
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num)), average_duration) + self.rng.randint(average_duration,
                                                                                                  size=self.num)
            elif range_max > self.num:
                offsets = np.sort(self.rng.randint(range_max, size=self.num))
            else:
                offsets = np.array(
                    list(range(range_max)) + [range_max - 1] * (self.num - range_max))
        else:
            if range_max > self.num:
                tick = range_max / float(self.num)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num)])
            else:
                offsets = np.array(
                    list(range(range_max)) + [range_max - 1] * (self.num - range_max))
        
        if index is not None:
            range_max = index.shape[0]
            return index[offsets]
        else:
            return offsets

            