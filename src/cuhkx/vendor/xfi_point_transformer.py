"""Selected X-Fi Point Transformer components (ICLR 2025, Apache-2.0).

Source: NTUMARS/X-Fi, commit 9ba74cc98beb2d75a8eadbe74b1b151dc3d39614.
Only unused allocator flushes and the overwritten random FPS seed are removed.
Public data uses [forward, left, up, SNR_dB, velocity], despite the original
(x, y, z, d, l) comment. See docs/PRETRAINED_RADAR_V16_ZH.md for evidence.
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1) # [B, S, [K]] -> [B, S*[K]]
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))
    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: mmwave pointcloud data, [B, N, 5]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    batch_indices = torch.arange(B, dtype=torch.long).to(device)

    barycenter = torch.mean(xyz, dim=1).view(B, 1, C)
    dist = torch.sum((xyz - barycenter) ** 2, -1)
    distance = torch.min(distance, dist)
    # print(torch.max(distance, -1)[0])
    farthest = torch.max(distance, -1)[1]

    for i in range(npoint):
        # print(farthest)
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        # print(torch.max(distance, -1)[0])
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False, knn=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]
    new_xyz = index_points(xyz, fps_idx)
    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
    else:
        idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample (k), C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, knn=self.knn)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0].transpose(1, 2)
        return new_xyz, new_points


class TransformerBlock(nn.Module):
    def __init__(self, d_points, d_model, k):
        super().__init__()
        self.fc1 = nn.Linear(d_points, d_model)
        self.fc2 = nn.Linear(d_model, d_points)
        self.fc_delta = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.fc_gamma = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        self.k = k

    # xyz: b x n x 3, features: b x n x f
    def forward(self, xyz, features):
        dists = square_distance(xyz, xyz) # b x n x n (self correlation result)
        knn_idx = dists.argsort()[:, :, :self.k]  # b x n x k (k nearest neighbors)
        knn_xyz = index_points(xyz, knn_idx) # b x n x k x 3 (n points, each point with its k nearest neighbors)

        pre = features
        x = self.fc1(features) # b x n x f -> b x n x d_model (f = d_points)
        q, k, v = self.w_qs(x), index_points(self.w_ks(x), knn_idx), index_points(self.w_vs(x), knn_idx)
        # q: b x n x d_model, k: b x n x k x d_model, v: b x n x k x d_model
        pos_enc = self.fc_delta(xyz[:, :, None] - knn_xyz)  # b x n x k x 3 -> b x n x k x d_model

        attn = self.fc_gamma(q[:, :, None] - k + pos_enc) # b x n x k x d_model
        attn = F.softmax(attn / np.sqrt(k.size(-1)), dim=-2)  # b x n x k x d_model

        res = torch.einsum('bmnf,bmnf->bmf', attn, v + pos_enc)
        res = self.fc2(res) + pre # b x n x d_model -> b x n x f
        return res, attn


class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)


class Transition(nn.Module):
    def __init__(self, in_channel, internal_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channel, internal_channel, 1),
            nn.BatchNorm1d(internal_channel),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(internal_channel, out_channel, 1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU()
        )
        # self.mlp_convs = nn.ModuleList()
        # self.mlp_bns = nn.ModuleList()
        # last_channel = in_channel
        # for out_channel in mlp: # mlp
        #     self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
        #     self.mlp_bns.append(nn.BatchNorm2d(out_channel))
        #     last_channel = out_channel

    def forward(self, xyz, features):
        """
        Input:
            xyz: input points position data, [B, N, 3]
            points: input points data, [B, N, D]
        Return:
            new_points_concat: sample points feature data, [B, N, 3+D]
        """
        new_features = torch.cat([xyz, features], dim=-1) # b x n x (3 + features)
        new_features = new_features.permute(0, 2, 1) # [B, 3+D, n]
        new_features = self.conv1(new_features) # [B, internal_channel, n]
        new_features = self.conv2(new_features) # [B, out_channel, n]
        new_features = new_features.permute(0, 2, 1) # [B, n, out_channel]
        return new_features


class Backbone(nn.Module):
    def __init__(self, n_points, nblocks, nneighbor, input_dim, transformer_dim):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32)
        )
        self.transformer1 = TransformerBlock(32, transformer_dim, nneighbor) # input channels = 32
        self.transition_downs = nn.ModuleList()
        self.transformers = nn.ModuleList()
        append_blocks = nblocks - 1
        self.transition_downs_1 = TransitionDown(n_points, nneighbor, [32+3, 64, 64])
        for i in range(append_blocks):
            channel = 32 * 2 ** (i+1)
            if i != 0:
                self.transition_downs.append(Transition(channel // 2 + 3, channel, channel))
            self.transformers.append(TransformerBlock(channel, transformer_dim, nneighbor))
        self.append_blocks = append_blocks

    def forward(self, x):
        # x: (B, N, 5). 5 = (x, y, z, d, l)
        xyz = x[..., :3]
        # print(f"xyz: {xyz.shape}")
        # print(f"x: {x.shape}")
        points = self.transformer1(xyz, self.fc1(x))[0]
        # print(f"points: {points.shape}")
        xyz, points = self.transition_downs_1(xyz, points)
        # print(f"points: {points.shape}")
        xyz_and_feats = [(xyz, points)]
        for i in range(self.append_blocks):
            if i == 0:
                points = self.transformers[i](xyz, points)[0]
                # print(f"points: {points.shape}")
                xyz_and_feats.append((xyz, points))
            else:
                points = self.transition_downs[i-1](xyz, points)
                # print(f"points: {points.shape}")
                points = self.transformers[i](xyz, points)[0]
                # print(f"points: {points.shape}")
                xyz_and_feats.append((xyz, points))
        return points, xyz_and_feats


class mmwave_PointTransformerReg(nn.Module):
    def __init__(self, input_dim = 5, n_points = 32, nblocks = 5, nneighbor = 16, transformer_dim = 128, n_classes = 27):
        super().__init__()
        self.backbone = Backbone(
            n_points= n_points,
            nblocks = nblocks,
            nneighbor = nneighbor,
            input_dim = input_dim,
            transformer_dim = transformer_dim
        )
        self.fc2 = nn.Sequential(
            nn.Linear(32 * 2 ** (nblocks-1), 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_classes)
        )
        self.nblocks = nblocks
        self.n_p = n_classes

    def forward(self, x):
        points, _ = self.backbone(x)
        res = self.fc2(points.mean(1))
        # pts = res.reshape(-1, self.n_p, 3)
        return res
