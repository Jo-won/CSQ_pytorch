import cv2
import numpy as np

import torch


class Compose(object):
    """Composes several video_transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> video_transforms.Compose([
        >>>     video_transforms.CenterCrop(10),
        >>>     video_transforms.ToTensor(),
        >>> ])
    """
    def __init__(self, transforms, aug_seed=0):
        self.transforms = transforms
        for i, t in enumerate(self.transforms):
            t.set_random_state(seed=(aug_seed+i))

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


class Transform(object):
    """basse class for all transformation"""
    def set_random_state(self, seed=None):
        self.rng = np.random.RandomState(seed)


class Normalize(Transform):
    """Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """
    def __init__(self, mean, std, group=False):
        self.mean = mean
        self.std = std
        self.group = group

    def __call__(self, tensor):
        if self.group is False:
            norm_tensor = self._process(tensor)
        else:
            norm_tensor=torch.Tensor([])
            for t in tensor:
                norm_t = self._process(t)
                norm_tensor = torch.cat((norm_tensor, norm_t.unsqueeze(0)), dim=0)
        return norm_tensor

    def _process(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

class Resize(Transform):
    """ Rescales the input numpy array to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: cv2.INTER_LINEAR
    """
    def __init__(self, size, interpolation=cv2.INTER_LINEAR, group=False):
        self.size = size # [w, h]
        self.interpolation = interpolation
        self.group = group

    def __call__(self, data):

        if self.group is False:
            scaled_data = self._process(data)
        else:
            scaled_data=[]
            for d in data:
                scaled_d = self._process(d)
                scaled_data.append(scaled_d)
            scaled_data = np.asarray(scaled_data)
        return scaled_data

    def _process(self, data):
        h, w, c = data.shape

        if isinstance(self.size, int):
            slen = self.size
            if min(w, h) == slen:
                return data
            if w < h:
                new_w = self.size
                new_h = int(self.size * h / w)
            else:
                new_w = int(self.size * w / h)
                new_h = self.size
        else:
            new_w = self.size[0]
            new_h = self.size[1]

        if (h != new_h) or (w != new_w):
            scaled_data = cv2.resize(data, (new_w, new_h), self.interpolation)
        else:
            scaled_data = data

        return scaled_data


class CenterCrop(Transform):
    """Crops the given numpy array at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """
    def __init__(self, size, group=False):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.group = group

    def __call__(self, data):

        if self.group is False:
            cropped_data = self._process(data)
        else:
            cropped_data=[]
            for d in data:
                cropped_d = self._process(d)
                cropped_data.append(cropped_d)
            cropped_data = np.asarray(cropped_data)
        
        return cropped_data

    def _process(self, data):

        h, w, c = data.shape
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        cropped_data = data[y1:(y1+th), x1:(x1+tw), :]

        return cropped_data

class ToTensor(Transform):
    """Converts a numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """
    def __init__(self, dim=3, group=False):
        self.dim = dim
        self.group = group

    def __call__(self, image):
        
        if self.group is False:
            tensor_data = self._process(image)
        else:
            tensor_data=torch.Tensor([])
            for i in image:
                tensor_d = self._process(i)
                tensor_data = torch.cat((tensor_data, tensor_d.unsqueeze(0)), dim=0)

        return tensor_data

    def _process(self, image):
        if isinstance(image, np.ndarray):
            # H, W, C = image.shape
            # handle numpy array
            image = torch.from_numpy(image.transpose((2, 0, 1)))
            # backward compatibility
            return image.float() / 255.0
            
class RandomHorizontalFlip(Transform):
    """Randomly horizontally flips the given numpy array with a probability of 0.5
    """
    def __init__(self, group=False):
        self.rng = np.random.RandomState(0)
        self.group = group

    def __call__(self, data):
        
        if self.rng.rand() < 0.5:
            if self.group is False:
                fliped_data = self._process(data)
            else:
                fliped_data=[]
                for d in data:
                    fliped_d = self._process(d)
                    fliped_data.append(fliped_d)
                fliped_data = np.asarray(fliped_data)
        else:
            fliped_data = data

        return fliped_data
    
    def _process(self, data):
        data = np.fliplr(data)
        data = np.ascontiguousarray(data)
        return data

class RandomHLS(Transform):
    def __init__(self, vars=[15, 35, 25], group=False):
        self.vars = vars
        self.rng = np.random.RandomState(0)
        self.group = group

    def __call__(self, data):
        
        if self.group is False:
            augmented_data = self._process(data)
        else:
            augmented_data=[]
            for d in data:
                augmented_d = self._process(d)
                augmented_data.append(augmented_d)
            augmented_data = np.asarray(augmented_data) 
        
        return augmented_data

    def _process(self, data):
        h, w, c = data.shape
        assert c%3 == 0, "input channel = %d, illegal"%c

        random_vars = [int(round(self.rng.uniform(-x, x))) for x in self.vars]

        base = len(random_vars)
        augmented_data = np.zeros(data.shape, )

        for i_im in range(0, int(c/3)):
            augmented_data[:,:,3*i_im:(3*i_im+3)] = \
                    cv2.cvtColor(data[:,:,3*i_im:(3*i_im+3)], cv2.COLOR_RGB2HLS)

        hls_limits = [180, 255, 255]
        for ic in range(0, c):
            var = random_vars[ic%base]
            limit = hls_limits[ic%base]
            augmented_data[:,:,ic] = np.minimum(np.maximum(augmented_data[:,:,ic] + var, 0), limit)

        for i_im in range(0, int(c/3)):
            augmented_data[:,:,3*i_im:(3*i_im+3)] = \
                    cv2.cvtColor(augmented_data[:,:,3*i_im:(3*i_im+3)].astype(np.uint8), \
                        cv2.COLOR_HLS2RGB)

        return augmented_data

class GroupMultiScaleCrop(Transform):

    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]

    def __call__(self, img_group):
        
        height, width, _ = img_group[0].shape
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(width, height)

        resize_fn = Resize((self.input_size[0], self.input_size[1]))
        crop_img_group = [img[offset_h:offset_h+crop_h, offset_w:offset_w+crop_w] for img in img_group]
        ret_img_group = [resize_fn(img) for img in crop_img_group]
        ret_img_group = np.asarray(ret_img_group)
        return ret_img_group

    def _sample_crop_size(self, image_w, image_h):
        # find a crop size
        
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))
        
        sind = np.random.randint(0,len(pairs))
        crop_pair = pairs[sind]
        if not self.fix_crop:
            w_offset = np.random.randint(0, image_w - crop_pair[0])
            h_offset = np.random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        sind = np.random.randint(0,len(offsets))
        return offsets[sind]

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret