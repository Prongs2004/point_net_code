'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, root, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.normal_channel = args.use_normals
        self.num_category = args.num_category
        self.use_normals = args.use_normals  # 添加这一行
        # 文件路径
        self.train_file = os.path.join(self.root, 'modelnet40_train.txt')
        self.test_file = os.path.join(self.root, 'modelnet40_test.txt')

        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')

        # 类别
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        #  关键：构造 datapath（官方标准）
        shape_ids = {}
        shape_ids['train'] = [line.rstrip() for line in open(self.train_file)]
        shape_ids['test'] = [line.rstrip() for line in open(self.test_file)]

        assert (split == 'train' or split == 'test')
        shape_names = [x.split('/')[0] for x in shape_ids[split]]

        self.datapath = []

        for item in shape_ids[split]:
            item = item.strip()  # e.g. night_stand_0001

            # ✅ 正确：从右边切一次
            cls = '_'.join(item.split('_')[:-1])  # night_stand
            name = item  # night_stand_0001

            self.datapath.append((cls, name))

        print(f"The size of {split} data is {len(self.datapath)}")

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        cls, name = self.datapath[index]
        label = self.classes[cls]

        # ===== npy路径 =====
        npy_path = os.path.join(self.root, cls, name + ".npy")

        # ===== txt路径（原始数据）=====
        txt_root = self.root.replace("modelnet40_preprocessed", "modelnet40_normal_resampled")
        txt_path = os.path.join(txt_root, cls, name + ".txt")

        # ===== 读取 =====
        if os.path.exists(npy_path):
            point_set = np.load(npy_path).astype(np.float32)
        elif os.path.exists(txt_path):
            point_set = np.loadtxt(txt_path, delimiter=',').astype(np.float32)
        else:
            raise FileNotFoundError(f"找不到: {npy_path} 或 {txt_path}")

        # ===== 采样 =====
        if len(point_set) >= self.npoints:
            choice = np.random.choice(len(point_set), self.npoints, replace=False)
        else:
            choice = np.random.choice(len(point_set), self.npoints, replace=True)

        point_set = point_set[choice, :]

        if not self.use_normals:
            point_set = point_set[:, 0:3]
        
        # ===== 生成 bbox =====
        xyz = point_set[:, 0:3]

        xyz_min = np.min(xyz, axis=0)
        xyz_max = np.max(xyz, axis=0)

        center = (xyz_min + xyz_max) / 2
        size = (xyz_max - xyz_min)

        bbox = np.concatenate([center, size]).astype(np.float32)
        return point_set, label, bbox

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
