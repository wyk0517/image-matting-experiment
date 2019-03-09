from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np
import cv2
import random
from utils import util
import torch.utils.data as data
import os
import torch
import torch.utils.data.sampler as sampler


def safe_crop(mat, x, y, crop_size=(320, 320)):
    crop_h, crop_w = crop_size
    if len(mat.shape) == 2:
        ret = np.zeros((crop_h, crop_w), dtype=np.float32)
    else:
        ret = np.zeros((crop_h, crop_w, 3), dtype=np.float32)  # HWC
    
    h = mat.shape[0]
    w = mat.shape[1]
    radio = w / h
    if h < crop_h:
        new_w = int(crop_h * radio)
        mat = cv2.resize(mat, dsize=(new_w, crop_h), interpolation=cv2.INTER_CUBIC)
    w = mat.shape[1]
    if w < crop_w:
        new_h = int(crop_w / radio)
        mat = cv2.resize(mat, dsize=(crop_w, new_h), interpolation=cv2.INTER_CUBIC)
    h = mat.shape[0]
    w = mat.shape[1]
    
    maxh = y + crop_h
    maxw = x + crop_w
    #print(crop_size)
    #print(h,w,maxh,maxw,x,y)
    if maxh > h:
        y = h - crop_h
    if maxw > w:
        x = w - crop_w


    crop_area = mat[y: y + crop_h, x:x + crop_w]  # don't exceed mat's bound
    h, w = crop_area.shape[:2]
    ret[0:h, 0:w] = crop_area

    if crop_size != (320, 320):
        ret = cv2.resize(ret, dsize=(320, 320), interpolation=cv2.INTER_CUBIC)  # finally, all resize to 320*320

    return ret


class my_Transform(object):
    def __init__(self, flip=False):
        self.flip = flip

    def __call__(self, img, alpha, fg, bg, trimap):
        different_sizes = [(320, 320), (480, 480), (640, 640)]
        crop_size = random.choice(different_sizes)
        crop_h, crop_w = crop_size
        # random crop in the unknown region center
        unknown_area = np.where(trimap == 128)
        unknown_area_num = len(unknown_area[0])
        x, y = 0, 0
        if unknown_area_num > 0:
            id = np.random.choice(range(unknown_area_num))
            center_x = unknown_area[1][id]
            center_y = unknown_area[0][id]
            x = max(0, center_x - int(crop_w / 2))
            y = max(0, center_y - int(crop_h / 2))

        img = safe_crop(img, x, y, crop_size)
        alpha = safe_crop(alpha, x, y, crop_size)
        bg = safe_crop(bg, x, y, crop_size)
        fg = safe_crop(fg, x, y, crop_size)
        trimap = safe_crop(trimap, x, y, crop_size)

        if self.flip and random.random() <= 0.5:
            img = cv2.flip(img, 1)
            alpha = cv2.flip(alpha, 1)
            fg = cv2.flip(fg, 1)
            bg = cv2.flip(bg, 1)
            trimap = cv2.flip(trimap, 1)

        return img, alpha, fg, bg, trimap


def get_files(dir):
    res = []
    for root, dirs, files in os.walk(dir, followlinks=True):
        for f in files:
            if f.endswith(".jpg") or f.endswith(".png") or f.endswith(".jpeg") or f.endswith(".JPG") or f.endswith(
                    '.PNG'):
                res.append(f)

    return res


def image_resize(image,shape=None):
    #h = image.shape[0]
    #w = image.shape[1]
    #new_h = h - h % 32
    #new_w = w - w % 32
    #if shape != None:
    #    new_h, new_w, _ = shape
    new_w = 640
    new_h = 640
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return image


class DIMdataset(data.Dataset):
    def __init__(self, config, usage, transform=None):
        self.transform = transform
        self.config = config
        self.samples = []
        self.usage = usage
        # fg_paths = get_files(config['fg_dir'])
        # bg_paths = get_files(config['bg_dir'])
        # image path
        self.image_paths = config['data_loader']['{}_image_dir'.format(usage)]
        self.fg_paths = config['data_loader']['{}_fg_dir'.format(usage)]
        self.bg_paths = config['data_loader']['{}_bg_dir'.format(usage)]
        self.alpha_paths = config['data_loader']['{}_alpha_dir'.format(usage)]
        # image name
        self.image_files = get_files(config['data_loader']['{}_image_dir'.format(usage)])
        fg_files = config['data_loader']['{}_fg_names'.format(usage)]
        bg_files = config['data_loader']['{}_bg_names'.format(usage)]
        with open(fg_files, 'r') as f:
            self.fg_names = f.read().splitlines()
        with open(bg_files, 'r') as f:
            self.bg_names = f.read().splitlines()

        # self.fg_cnt = len(fg_paths)
        # self.bg_cnt = len(bg_paths)
        self.image_cnt = len(self.image_paths)
        """
        for fg_path in fg_paths:
            alpha_path = fg_path.replace(config['fg_dir'], config['alpha_dir'])  # because alpha_name = fg_name only the  dir is diff
            assert(os.path.exists(alpha_path))
            assert(os.path.exists(fg_path))
            self.fg_samples.append(alpha_path, fg_path)
        print("\t--Valid FG Samples:{}" .format(self.fg_cnt))

        for bg_path in bg_paths:
            assert(os.path.exists(bg_path))
            self.bg_samples.append(bg_path)
        print("\t -- Vaild BG Samples:{}" .format(self.bg_cnt))

        assert(self.fg_cnt > 0 and self.bg_cnt > 0)
        """
        """        
        for fg_path in fg_paths:
            alpha_path = fg_path.replace(self.args.fgDir, self.args.alphaDir)
            img_path = fg_path.replace(self.args.fgDir, self.args.imgDir)
            bg_path = fg_path.replace(self.args.fgDir, self.args.bgDir)
            assert(os.path.exists(alpha_path))
            assert(os.path.exists(fg_path))
            assert(os.path.exists(bg_path))
            assert(os.path.exists(img_path))
            self.samples.append((alpha_path, fg_path, bg_path, img_path))
        print("\t--Valid Samples: {}".format(self.cnt))
        assert(self.cnt > 0)
        """

        for image_name in self.image_files:
            fg_index = int(image_name.split('_')[0])
            bg_index = int(image_name.split('_')[1].split('.')[0])
            fg_name = self.fg_names[fg_index]
            bg_name = self.bg_names[bg_index]
            fg_path = os.path.join(self.fg_paths, fg_name)
            alpha_path = os.path.join(self.alpha_paths, fg_name)  # alpha has the same name as fg
            bg_path = os.path.join(self.bg_paths, bg_name)
            image_path = os.path.join(self.image_paths, image_name)
            assert (os.path.exists(alpha_path))
            assert (os.path.exists(fg_path))
            assert (os.path.exists(bg_path))
            assert (os.path.exists(image_path))
            self.samples.append((alpha_path, fg_path, bg_path, image_path))
        print("\t--Valid Samples: {}".format(len(self.samples)))
        assert (self.image_cnt > 0)
        # super(DIMDataLoader, self).__init__(self.samples, batch_size, shuffle, validation_split,
        #                                     num_workers)

    def __getitem__(self, item):
        alpha_path, fg_path, bg_path, img_path = self.samples[item]

        img_info = [fg_path, alpha_path, bg_path, img_path]
        # read fg, alpha
        fg = cv2.imread(fg_path)[:, :, 0:3]
        alpha = cv2.imread(alpha_path)
        alpha = cv2.cvtColor(alpha, cv2.COLOR_RGB2GRAY)  # get a 2-dim  [:, :, 0]  # get a 2-dim
        bg = cv2.imread(bg_path)[:, :, 0:3]
        img = cv2.imread(img_path)[:, :, 0:3]

        if self.usage == 'test':
            img = image_resize(img)
            fg = image_resize(fg)
            bg = image_resize(bg,img.shape)
            alpha = image_resize(alpha)
        
        # print("read: bg:{},fg:{},img:{}" .format(bg.shape, fg.shape, img.shape))
        # assert(bg.shape == fg.shape and bg.shape == img.shape)
        img_info.append(fg.shape)
        h, w, c = fg.shape  # shape is HWC

        trimap = util.gen_trimap(alpha)

        # random crop and flip
        if self.usage != "test" and self.transform:
        #if self.transform:
            img, alpha, fg, bg, trimap = self.transform(img, alpha, fg, bg, trimap)

        trimap = util.gen_trimap(alpha)
        # grad = util.compute_gradient(img)

        # change numpy to tensor and change [HWC] -> [CHW]
        # print("alpha:{}, trimap:{}, bg:{},fg:{},img:{}".format(alpha.shape, trimap.shape, bg.shape, fg.shape, img.shape))
        # input()
        alpha = torch.from_numpy(alpha.astype(np.float32)[np.newaxis, :, :])
        trimap = torch.from_numpy(trimap.astype(np.float32)[np.newaxis, :, :])
        # grad = torch.from_numpy(grad.astype(np.float32)[np.newaxis, :, :])
        img = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
        fg = torch.from_numpy(fg.astype(np.float32)).permute(2, 0, 1)
        bg = torch.from_numpy(bg.astype(np.float32)).permute(2, 0, 1)
        return img, alpha, fg, bg, trimap

    def __len__(self):
        return len(self.samples)


def dataloader(config, usage, batch_size, shuffle, validation_split, num_workers, transform_switch):
    if transform_switch is True:
        transform = my_Transform(flip=True)
        dataset = DIMdataset(config, usage, transform)
    else:
        dataset = DIMdataset(config, usage)
    num_samples = len(dataset)
    # print(num_samples)
    idx_full = np.arange(num_samples)  # 生成一个1-n的list

    np.random.seed(0)  # 随机种子(numpy)
    np.random.shuffle(idx_full)  # 随机

    len_valid = int(num_samples * validation_split)  # 验证数据集大小

    valid_idx = idx_full[0:len_valid]  # 获取验证样本
    train_idx = np.delete(idx_full, np.arange(0, len_valid))  # 把验证集的序号删掉
    train_sampler = sampler.SubsetRandomSampler(train_idx)  # 获取训练样本
    valid_sampler = sampler.SubsetRandomSampler(valid_idx)
    # print(len(train_sampler))
    # print(len(valid_sampler))
    train_data_loader = data.DataLoader(dataset=dataset, sampler=train_sampler, batch_size=batch_size,
                                        num_workers=num_workers)
    valid_data_loader = data.DataLoader(dataset=dataset, sampler=valid_sampler, batch_size=batch_size,
                                        num_workers=num_workers)
    # print(len(train_data_loader))
    # print(len(valid_data_loader))
    return train_data_loader, valid_data_loader


def dataloader_test(config, usage, batch_size, shuffle, validation_split, num_workers, transform_switch):
    if transform_switch is True:
        transform = my_Transform(flip=True)
        dataset = DIMdataset(config, usage, transform)
    else:
        dataset = DIMdataset(config, usage)
    if usage == 'train':  
        data_loader = data.DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size,
                                      num_workers=num_workers)
    else:
        data_loader = data.DataLoader(dataset=dataset, shuffle=shuffle, batch_size=batch_size,
                                      num_workers=num_workers)
    # print(len(train_data_loader))
    # print(len(valid_data_loader))
    return data_loader
