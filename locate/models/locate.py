import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dino import vision_transformer as vits
from models.dino.utils import load_pretrained_weights
from models.model_util import *
from fast_pytorch_kmeans import KMeans
import cv2
import numpy as np
from utils.util import  normalize_map

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Net(nn.Module):

    def __init__(self, aff_classes=36):
        super(Net, self).__init__()

        self.aff_classes = aff_classes
        self.gap = nn.AdaptiveAvgPool2d(1)

        # --- hyper-parameters --- #
        self.aff_cam_thd = 0.6
        self.part_iou_thd = 0.6
        self.cel_margin = 0.5

        # --- dino-vit features --- #
        self.vit_feat_dim = 384
        self.cluster_num = 3
        self.stride = 16
        self.patch = 16

        self.vit_model = vits.__dict__['vit_small'](patch_size=self.patch, num_classes=0)
        load_pretrained_weights(self.vit_model, '', None, 'vit_small', self.patch)

        # --- learning parameters --- #
        self.aff_proj = Mlp(in_features=self.vit_feat_dim, hidden_features=int(self.vit_feat_dim * 4),
                            act_layer=nn.GELU, drop=0.)
        self.aff_ego_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_exo_proj = nn.Sequential(
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
            nn.Conv2d(self.vit_feat_dim, self.vit_feat_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.vit_feat_dim),
            nn.ReLU(True),
        )
        self.aff_fc = nn.Conv2d(self.vit_feat_dim, self.aff_classes, 1)

    def forward(self, exo, ego, aff_label, epoch):

        num_exo = exo.shape[1]  # 3
        exo = exo.flatten(0, 1)  # b*num_exo x 3 x 224 x 224  把exo的图像都压缩到第一维度上去

        # --- Extract deep descriptors from DINO-vit --- #
        with torch.no_grad():                                        # ego: b x 3 x 224 x 224
            _, ego_key, ego_attn = self.vit_model.get_last_key(ego)  # ego_attn: b x 6 x (1+hw) x (1+hw) 最后一层的特征图的大小为14*14 exo_key: b x 6 x (1+hw) x 64
            _, exo_key, exo_attn = self.vit_model.get_last_key(exo)  # exo_attn: bxnum_exo x 6 x (1+hw) x (1+hw)  exo_key: bxnum_exo x 6 x (1+hw) x 64
            ego_desc = ego_key.permute(0, 2, 3, 1).flatten(-2, -1).detach()  # b x (1+hw) x 64*6 就是将变形后的最后面两个维度压缩到一块儿
            exo_desc = exo_key.permute(0, 2, 3, 1).flatten(-2, -1).detach()  # b*num x (1+hw) x 64*6

        ego_proj = ego_desc[:, 1:] + self.aff_proj(ego_desc[:, 1:]) # b x (hw) x 64*6 去除cls这个token
        exo_proj = exo_desc[:, 1:] + self.aff_proj(exo_desc[:, 1:]) # b*num x (hw) x 64*6
        ego_desc = self._reshape_transform(ego_desc[:, 1:, :], self.patch, self.stride)  # b x 384 x h x w 就是进行了一个变换操作
        exo_desc = self._reshape_transform(exo_desc[:, 1:, :], self.patch, self.stride)  # b*num x 384 x h x w
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)  # b x 384 x h x w
        exo_proj = self._reshape_transform(exo_proj, self.patch, self.stride)  # b x 384 x h x w  这上下两个维度一样，有什么作用吗？

        b, c, h, w = ego_desc.shape  # [16,384,14,14]
        ego_cls_attn = ego_attn[:, :, 0, 1:].reshape(b, 6, h, w) # [16,6,1,196]-->[16,6,14,14] 取出cls的注意力分数
        ego_cls_attn = (ego_cls_attn > ego_cls_attn.flatten(-2, -1).mean(-1, keepdim=True).unsqueeze(-1)).float() # [16,6,14,14] 过程，主要是将连续点转化为0或者1的值 [16,6,14,14]-->[16,6,196] --> 然后对196进行均值化，之后就变为[16,6,1]之后再增加一个维度[16,6,1,1]
        head_idxs = [0, 1, 3] # 这一块儿为什么要去0,1,3呢，
        ego_sam = ego_cls_attn[:, head_idxs].mean(1) #[16,14,14] 取出来的值进行均值化，其他的就不要了？
        ego_sam = normalize_minmax(ego_sam)
        ego_sam_flat = ego_sam.flatten(-2, -1) # [16,196]

        # --- Affordance CAM generation --- #
        exo_proj = self.aff_exo_proj(exo_proj)  # b*num_exo x 384 x h x w
        aff_cam = self.aff_fc(exo_proj)  # b*num_exo x 25 x h x w
        aff_logits = self.gap(aff_cam).reshape(b, num_exo, self.aff_classes) # [16,3,25]
        aff_cam_re = aff_cam.reshape(b, num_exo, self.aff_classes, h, w) # [16,3,25,14,14]

        gt_aff_cam = torch.zeros(b, num_exo, h, w).cuda() # [16,3,14,14]
        for b_ in range(b):
            gt_aff_cam[b_, :] = aff_cam_re[b_, :, aff_label[b_]]
            # cam可视化
            #viz_cam = aff_cam_re[b_, :, aff_label[b_]]  # 3*14*14
            # viz_cam = torch.squeeze(torch.mean(aff_cam_re[b_, :, aff_label[b_]],dim=0,keepdim=True)) # 3*14*14
            # viz_each_feature = viz_cam.detach().cpu().numpy()*255
            # viz_each_feature_to_color = cv2.applyColorMap(cv2.resize(np.array(viz_each_feature, np.uint8),(224,224)), cv2.COLORMAP_JET)
            # cv2.imwrite('./viz.png',viz_each_feature_to_color)

            # for index, viz in enumerate(viz_cam):
            #     viz_each_feature = viz.detach().cpu().numpy()*255
            #     viz_each_feature_to_color = cv2.applyColorMap(np.array(viz_each_feature, np.uint8), cv2.COLORMAP_JET)
            #     cv2.imwrite('./{0}.png'.format(index),viz_each_feature_to_color)

        # --- Clustering extracted descriptors based on CAM --- #
        ego_desc_flat = ego_desc.flatten(-2, -1)  # 16 x 384 x hw
        exo_desc_re_flat = exo_desc.reshape(b, num_exo, c, h, w).flatten(-2, -1) # [16,3,384,196]
        sim_maps = torch.zeros(b, self.cluster_num, h * w).cuda()  # [16,3,196]
        exo_sim_maps = torch.zeros(b, num_exo, self.cluster_num, h * w).cuda()  # [16,3,3,196]
        part_score = torch.zeros(b, self.cluster_num).cuda()  #[16,3]
        part_proto = torch.zeros(b, c).cuda() # [16,384]
        for b_ in range(b):
            exo_aff_desc = []
            for n in range(num_exo):
                tmp_cam = gt_aff_cam[b_, n].reshape(-1)  # [196]
                tmp_max, tmp_min = tmp_cam.max(), tmp_cam.min()
                tmp_cam = (tmp_cam - tmp_min) / (tmp_max - tmp_min + 1e-10) # [196]
                tmp_desc = exo_desc_re_flat[b_, n] # [384,196]
                tmp_top_desc = tmp_desc[:, torch.where(tmp_cam > self.aff_cam_thd)[0]].T   # n x c  [74,384]
                exo_aff_desc.append(tmp_top_desc) #
            exo_aff_desc = torch.cat(exo_aff_desc, dim=0)  # (n1 + n2 + n3) x c [164,384]

            if exo_aff_desc.shape[0] < self.cluster_num:
                continue

            kmeans = KMeans(n_clusters=self.cluster_num, mode='euclidean', max_iter=300)
            kmeans.fit_predict(exo_aff_desc.contiguous())
            clu_cens = F.normalize(kmeans.centroids, dim=1) # [3.384]

            # save the exocentric similarity maps for visualization in training
            for n_ in range(num_exo):
                exo_sim_maps[b_, n_] = torch.mm(clu_cens, F.normalize(exo_desc_re_flat[b_, n_], dim=0))

            # find object part prototypes and background prototypes
            sim_map = torch.mm(clu_cens, F.normalize(ego_desc_flat[b_], dim=0))  # self.cluster_num x hw [3,196]
            tmp_sim_max, tmp_sim_min = torch.max(sim_map, dim=-1, keepdim=True)[0], \
                                       torch.min(sim_map, dim=-1, keepdim=True)[0]
            sim_map_norm = (sim_map - tmp_sim_min) / (tmp_sim_max - tmp_sim_min + 1e-12)

            sim_map_hard = (sim_map_norm > torch.mean(sim_map_norm, 1, keepdim=True)).float() # [3,196]
            sam_hard = (ego_sam_flat > torch.mean(ego_sam_flat, 1, keepdim=True)).float() # [16,196]

            inter = (sim_map_hard * sam_hard[b_]).sum(1) # [3]
            union = sim_map_hard.sum(1) + sam_hard[b_].sum() - inter  # [3]
            p_score = (inter / sim_map_hard.sum(1) + sam_hard[b_].sum() / union) / 2  # [3]

            sim_maps[b_] = sim_map
            part_score[b_] = p_score

            if p_score.max() < self.part_iou_thd:
                continue

            part_proto[b_] = clu_cens[torch.argmax(p_score)]

        sim_maps = sim_maps.reshape(b, self.cluster_num, h, w)  # [3,196]
        exo_sim_maps = exo_sim_maps.reshape(b, num_exo, self.cluster_num, h, w)  # [16,3,3,14,14]
        ego_proj = self.aff_ego_proj(ego_proj)  # [16,384,14,14]
        ego_pred = self.aff_fc(ego_proj)  #[16,25,14,14]
        aff_logits_ego = self.gap(ego_pred).view(b, self.aff_classes) # [16,25]

        # --- concentration loss --- #
        gt_ego_cam = torch.zeros(b, h, w).cuda()  #[16,14,14]
        loss_con = torch.zeros(1).cuda()  # [1]
        for b_ in range(b):
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]
            loss_con += concentration_loss(ego_pred[b_])

        gt_ego_cam = normalize_minmax(gt_ego_cam)  # [16,14,14]
        loss_con /= b

        # --- prototype guidance loss --- #
        loss_proto = torch.zeros(1).cuda()
        valid_batch = 0
        if epoch[0] > epoch[1]:
            for b_ in range(b):
                if not part_proto[b_].equal(torch.zeros(c).cuda()):
                    mask = gt_ego_cam[b_]  # [14,14]
                    tmp_feat = ego_desc[b_] * mask  # [384,14,14]
                    embedding = tmp_feat.reshape(tmp_feat.shape[0], -1).sum(1) / mask.sum() # [384]
                    loss_proto += torch.max(
                        1 - F.cosine_similarity(embedding, part_proto[b_], dim=0) - self.cel_margin,
                        torch.zeros(1).cuda()) #这一块儿对应这途中的公式5
                    valid_batch += 1
            loss_proto = loss_proto / (valid_batch + 1e-15)

        masks = {'exo_aff': gt_aff_cam, 'ego_sam': ego_sam,
                 'pred': (sim_maps, exo_sim_maps, part_score, gt_ego_cam)}
        logits = {'aff': aff_logits, 'aff_ego': aff_logits_ego}

        return masks, logits, loss_proto, loss_con

    @torch.no_grad()
    def test_forward(self, ego, aff_label):
        _, ego_key, ego_attn = self.vit_model.get_last_key(ego)  # attn: b x 6 x (1+hw) x (1+hw)
        ego_desc = ego_key.permute(0, 2, 3, 1).flatten(-2, -1)
        ego_proj = ego_desc[:, 1:] + self.aff_proj(ego_desc[:, 1:])
        ego_desc = self._reshape_transform(ego_desc[:, 1:, :], self.patch, self.stride)
        ego_proj = self._reshape_transform(ego_proj, self.patch, self.stride)

        b, c, h, w = ego_desc.shape
        ego_proj = self.aff_ego_proj(ego_proj)
        ego_pred = self.aff_fc(ego_proj)

        gt_ego_cam = torch.zeros(b, h, w).cuda()
        for b_ in range(b):
            gt_ego_cam[b_] = ego_pred[b_, aff_label[b_]]
            # viz_cam = torch.squeeze(torch.mean(ego_pred[b_, :, aff_label[b_]],dim=0,keepdim=True)) # 3*14*14
            # viz_cam = torch.squeeze(torch.mean(ego_pred[b_, :, aff_label[b_]],dim=0,keepdim=True)) # 3*14*14
            # viz_cam = torch.squeeze(gt_ego_cam) # 3*14*14
            # # viz_each_feature = viz_cam.detach().cpu().numpy()*255
            # viz_each_feature = viz_cam.detach().cpu().numpy()*255
            # # viz_each_feature = normalize_map(viz_each_feature, 224)
            # viz_each_feature_to_color = cv2.applyColorMap(cv2.resize(np.array(viz_each_feature, np.uint8),(224,224)), cv2.COLORMAP_JET)
            # cv2.imwrite('./viz_test.png',viz_each_feature_to_color)

        # return gt_ego_cam, ego_pred
        return gt_ego_cam

    def _reshape_transform(self, tensor, patch_size, stride):
        height = (224 - patch_size) // stride + 1
        width = (224 - patch_size) // stride + 1
        result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
        result = result.transpose(2, 3).transpose(1, 2).contiguous()
        return result
