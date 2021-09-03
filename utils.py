import os
import shutil
import torch.nn as nn
import torch
from torch.nn import functional as F
from PIL import Image
import numpy as np
import random
import numbers
import torchvision


def gaussian(x, mu, sigma):
    """ evaluate gaussian function for parameter x """
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - mu) ** 2 / (sigma ** 2))


def rbf_activations(x, mus, sigmas):
    """ normalized RBF activations for all values in image. Gaussian are parameterized by the mus / sigmas arrays """
    x = np.array(x)
    x = np.expand_dims(x, -1)
    activations = gaussian(x, np.array(mus), np.array(sigmas))
    if np.all(activations == 0):
        ret = np.zeros(shape=activations.shape, dtype=np.float)
        ret[np.argmin((x - mus) * (x - mus))] = 1
        return ret
    if activations.ndim == 1:
        activations /= np.sum(activations)
    else:
        sums = np.sum(activations, axis=-1)
        sums[sums == 0] = 1
        activations /= np.expand_dims(sums, -1)
    return activations


def make_paths_absolute(args):
    if "~" in args.data:
        args.data = os.path.expanduser(args.data)
    if hasattr(args, "save_model_path") and "~" in args.save_model_path:
        args.save_model_path = os.path.expanduser(args.save_model_path)
    if hasattr(args, "pretrained_model_path") and args.pretrained_model_path is not None and "~" in args.pretrained_model_path:
        args.pretrained_model_path = os.path.expanduser(args.pretrained_model_path)
    if hasattr(args, "save_model_path") and not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path, exist_ok=True)


def read_categories_from_dataset(args):
    """ read a file with one segmentation category/class per line """
    if os.path.exists(os.path.join(args.data, "categories")):
        if hasattr(args, "save_model_path"):
            shutil.copy(os.path.join(args.data, "categories"), os.path.join(args.save_model_path, "categories"))
        with open(os.path.join(args.data, "categories")) as f:
            categories = f.read().split("\n")
            categories = [c for c in categories if c.strip() != ""]
            args.num_classes = len(categories)
    else:
        categories = ["unknown{}".format(i) for i in range(args.num_classes)]
        print("NO CATEGORIES FILE IN DATASET ://")
        import time
        time.sleep(1)
    return categories


def write_categories(target_path, categories):
    """ write a file with one segmentation category/class per line """
    if os.path.exists(target_path) and os.path.isdir(target_path):
        target_path = os.path.join(target_path, "categories")
    else:
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
    with open(target_path, "w+") as f:
        f.write("\n".join(categories))


def load_matching_weights(model, pretrained_model_path):
    """ load weights from path into the given model if the name and the tensor shape matches """
    print('load model from %s ...' % pretrained_model_path)
    loaded_state_dict = torch.load(pretrained_model_path)
    if hasattr(loaded_state_dict, "state_dict"):
        loaded_state_dict = loaded_state_dict.state_dict()
    own_state = model.state_dict()
    restored_parameter_names = []
    for name_, param in loaded_state_dict.items():
        found = False
        shape_mismatch = None, None
        for name in [name_, "module." + name_, "module.encoder." + name_, "module.module." + name_, "module.module.encoder." + name_]:
            if name not in own_state or own_state[name] is None:
                continue
            if param.shape != own_state[name].shape:
                shape_mismatch = own_state[name].shape, param.shape, name
                continue
            own_state[name].copy_(param)
            found = True
            break
        if found:
            restored_parameter_names.append(name)
        else:
            if shape_mismatch[0] is not None:
                details = " own vs restored: {} vs {}".format(shape_mismatch[0], shape_mismatch[1])
                name = shape_mismatch[2]
            else:
                details = ' not in own model'
                name = name_
            print("[restore] failure: " + name + details)
    print('[restore] match success rate: {:.1f}%'.format(100. * float(len(restored_parameter_names)) / len(loaded_state_dict.keys())))
    return restored_parameter_names


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


def one_hot(label, num_classes):
    """ convert a index-encoded label image to a logits (C, H, W) tensor in numpy """
    n, m = label.shape
    logits = np.zeros((n * m, num_classes))  # empty, flat array
    logits[np.arange(n * m), label.flatten()] = 1  # one-hot encoding for 1D
    logits = logits.reshape((n, m, num_classes))  # reshaping back to 3D tensor
    # (H, W, C) to (C, H, W)
    logits = np.swapaxes(logits, 0, 2)
    logits = np.swapaxes(logits, 1, 2)
    return logits


def one_hot_torch(target, num_classes):
    """ convert a index-encoded label image to a logits (C, H, W) tensor using pytorch """
    target = target.unsqueeze(1)
    target_one_hot = torch.zeros((target.shape[0], num_classes, *target.shape[-2:]), dtype=torch.float32, device=target.device)
    ones = torch.ones(target_one_hot.size(), dtype=torch.float32, device=target.device)
    target_one_hot.scatter_(1, target, ones)
    return target_one_hot


def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.

    # Arguments
        image: The one-hot format image

    # Returns
        A 2D array with the same width and height as the input, but
        with a depth size of 1, where each pixel value is the classified
        class key.
    """
    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,1])

    # for i in range(0, w):
    #     for j in range(0, h):
    #         index, value = max(enumerate(image[i, j, :]), key=operator.itemgetter(1))
    #         x[i, j] = index
    image = image.permute(1, 2, 0)
    x = torch.argmax(image, dim=-1)
    return x


def reverse_one_hot_uint8(image):
    image = image.permute(1, 2, 0)
    x = torch.argmax(image, dim=-1).to(torch.uint8)
    return x


def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.

    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """

    # w = image.shape[0]
    # h = image.shape[1]
    # x = np.zeros([w,h,3])
    # colour_codes = label_values
    # for i in range(0, w):
    #     for j in range(0, h):
    #         x[i, j, :] = colour_codes[int(image[i, j])]
    label_values = [label_values[key][:3] for key in label_values if label_values[key][3] == 1]
    label_values.append([0, 0, 0])
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


def compute_precision(pred, label):
    pred = pred.flatten()
    label = label.flatten()
    total = len(label)
    return (pred == label).astype(np.float).sum() / float(total)


def fast_hist(a, b, n):
    '''
    a and b are predict and mask respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def compute_precision_torch(pred, label):
    # type: (torch.Tensor, torch.Tensor) -> float
    pred = pred.flatten()
    label = label.flatten()
    return ((pred == label).to(torch.float32).sum().cpu() / label.numel()).numpy()


def fast_hist_torch(a, b, n):
    '''
    a and b are predict and mask respectively
    n is the number of classes
    '''
    return torch.bincount(n * a.to(torch.int32) + b, minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    epsilon = 1e-5
    return (np.diag(hist) + epsilon) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


def downsample_1d_np(values, num_points):
    values = np.interp(np.linspace(0, len(values) - 1, num=num_points, endpoint=True),
                       np.arange(len(values)), values)
    return values


class RandomCrop(object):
    """Crop the given PIL Image at a random location.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception.
    """

    def __init__(self, size, seed, padding=0, pad_if_needed=False):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.seed = seed

    @staticmethod
    def get_params(img, output_size, seed):
        """Get parameters for ``crop`` for a random crop.

        Args:
            img (PIL Image): Image to be cropped.
            output_size (tuple): Expected output size of the crop.

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        random.seed(seed)
        w, h = img.size
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w
        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if self.padding > 0:
            img = torchvision.transforms.functional.pad(img, self.padding)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = torchvision.transforms.functional.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = torchvision.transforms.functional.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size, self.seed)

        return torchvision.transforms.functional.crop(img, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


def group_weight(weight_group, module, norm_layer, lr):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, norm_layer) or isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(
        group_no_decay)
    weight_group.append(dict(params=group_decay, lr=lr))
    weight_group.append(dict(params=group_no_decay, weight_decay=.0, lr=lr))
    return weight_group


if __name__ == '__main__':
    import cv2

    logits = one_hot(np.array([[0, 1], [2, 1]]), 4)
    cv2.imshow("logits class 1", logits[1, :, :])
    cv2.waitKey(0)
