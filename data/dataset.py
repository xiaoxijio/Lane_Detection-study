import torch
from PIL import Image
import os
import pdb
import numpy as np
import cv2
from data.mytransforms import find_start_pos


def loader_func(path):
    return Image.open(path)


class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane

    def __getitem__(self, index):
        name = self.list[index].split()[0]
        img_path = os.path.join(self.path, name)
        img = loader_func(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, name

    def __len__(self):
        return len(self.list)


class LaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None, target_transform=None, simu_transform=None, griding_num=50,
                 load_name=False,
                 row_anchor=None, use_aux=False, segment_transform=None, num_lanes=4):
        super(LaneClsDataset, self).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform
        self.path = path
        self.griding_num = griding_num
        self.load_name = load_name
        self.use_aux = use_aux
        self.num_lanes = num_lanes

        with open(list_path, 'r') as f:
            self.list = f.readlines()

        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def __getitem__(self, index):
        l = self.list[index]  # 数据路径和标签路径
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        label_path = os.path.join(self.path, label_name)
        label = loader_func(label_path)

        img_path = os.path.join(self.path, img_name)
        img = loader_func(img_path)

        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)  # 数据增强
        lane_pts = self._get_index(label)
        # get the coordinates of lanes at row anchors

        w, h = img.size
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)  # 相当于将lane_pts映射到griding_num个格子里
        # make the coordinates to classification label
        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)

        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.use_aux:
            return img, cls_label, seg_label
        if self.load_name:
            return img, cls_label, img_name
        return img, cls_label

    def __len__(self):
        return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape
        col_sample = np.linspace(0, w - 1, num_cols)  # 将宽分为num_cols个格子

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):  # 遍历每个车道线
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _get_index(self, label):
        w, h = label.size

        if h != 288:  # 因为先验值是基于高为288搞的，所以要将先验值映射到原图高度上
            scale_f = lambda x: int((x * 1.0 / 288) * h)
            sample_tmp = list(map(scale_f, self.row_anchor))  # 相当于一个先验值 18个车道点

        all_idx = np.zeros((self.num_lanes, len(sample_tmp), 2))  # (4, 18, 2) 4条车道线 18个车道点 2个行列位置 列为-1表示不存在
        for i, r in enumerate(sample_tmp):  # 遍历每个点 18
            label_r = np.asarray(label)[int(round(r))]  # 读取标签在这一行的所有信息
            for lane_idx in range(1, self.num_lanes + 1):  # 遍历每条车道线 4
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r  # 第几行
                    all_idx[lane_idx - 1, i, 1] = -1  # -1为没有信息
                    continue
                pos = np.mean(pos)  # 车道线是有宽度的 所以取个平均
                all_idx[lane_idx - 1, i, 0] = r  # 第几行
                all_idx[lane_idx - 1, i, 1] = pos  # 当前车道线列位置

        # data augmentation: extend the lane to the boundary of image

        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i, :, 1] == -1):  # 判断车道存不存在
                continue
            # if there is no lane
            # 对4条车道进行一个延伸，如果不存在这条车道就不
            valid = all_idx_cp[i, :, 1] != -1  # 第i个车道线的18个车道点是否存在
            # get all valid lane points' index
            valid_idx = all_idx_cp[i, valid, :]  # 存在的点坐标
            # get all valid lane points
            if valid_idx[-1, 0] == all_idx_cp[0, -1, 0]:
                # 如果最后一个有效车道点的y坐标已经是所有行的最后一个y坐标
                # 这意味着这条线已经达到了图像的底部边界
                # 所以我们跳过
                continue
            if len(valid_idx) < 6:
                continue
            # 如果车道太短，无法延伸

            valid_idx_half = valid_idx[len(valid_idx) // 2:, :]  # 取后半段的车道点
            p = np.polyfit(valid_idx_half[:, 0], valid_idx_half[:, 1], deg=1)  # 线性拟合 延长这条线到边界
            start_line = valid_idx_half[-1, 0]  # 取后半段车道点中最后一个点的 x 坐标作为起点，用于后续查找和拟合
            pos = find_start_pos(all_idx_cp[i, :, 0], start_line) + 1  # 找到 start_line 的索引位置

            fitted = np.polyval(p, all_idx_cp[i, pos:, 0])  # 得到拟合值
            fitted = np.array([-1 if y < 0 or y > w - 1 else y for y in fitted])  # 判断拟合值能否放进图中 -1表示无效

            assert np.all(all_idx_cp[i, pos:, 1] == -1)
            all_idx_cp[i, pos:, 1] = fitted  # 放进下一个位置
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp
