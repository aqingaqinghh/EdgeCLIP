import model.resnet as resnet
import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
import clip

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

class Attention(nn.Module):
    """
    Guided Attention Module (GAM).

    Args:
        in_channels: interval channel depth for both input and output
            feature map.
        drop_rate: dropout rate.
    """

    def __init__(self, in_channels, drop_rate=0.1):
        super().__init__()
        self.DEPTH = in_channels
        self.DROP_RATE = drop_rate
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.DEPTH, out_channels=self.DEPTH,
                      kernel_size=1),
            nn.Dropout(p=drop_rate),
            nn.Sigmoid())

    @staticmethod
    def mask(embedding, mask):
        h, w = embedding.size()[-2:]

        mask = F.interpolate(mask.unsqueeze(1), size=(h, w), mode='nearest')
        mask=mask
        return mask * embedding

    def forward(self, *x):
        if len(x) == 2:
            Fs, Ys = x
            att = F.adaptive_avg_pool2d(self.mask(Fs, Ys), output_size=(1, 1))
        else:
            Fs = x[0]
            att = F.adaptive_avg_pool2d(Fs, output_size=(1, 1))
        g = self.gate(att)
        Fs = g * Fs
        return Fs

def get_triple_loss(anchors, local_features, mask_pos, mask_neg):
    """
    Args:
        anchors: prototypes: [b, n_class, c]
        local_features:  [b, c, h, w]
        mask: [b,1,h,w]
        class_chosen: [b, ]  int

    Returns: triplet_loss
    """
    b, c, h, w = local_features.shape
    mask_pos = F.interpolate(mask_pos.float(), size=(h, w), mode="nearest").long().view(b, -1)  # [b, h*w]
    mask_neg = F.interpolate(mask_neg.float(), size=(h, w), mode="nearest").long().view(b, -1)  # [b, h*w]
    local_features = local_features.view(b, c, -1).permute(0, 2, 1).contiguous() # [b, h*w, c]
    triplet_loss = torch.Tensor([0.0]).cuda()
    length = 1   # number of triplets

    # hard triplet dig
    count = b
    for i in range(b):
        anchor_list = []
        mask_i_pos = mask_pos[i]  # [h*w]
        mask_i_neg = mask_neg[i]  # [h*w]
        negative_list_i = local_features[i][mask_i_neg == 1]
        positive_list_i = local_features[i][mask_i_pos == 1] #N,C
        if positive_list_i.shape[0] <length or negative_list_i.shape[0]< length:
            temp_loss = torch.Tensor([0.0]).cuda()
            count = count - 1
        else:
            temp_loss = hard_triplet_dig(anchors[i].unsqueeze(0), positive_list_i, negative_list_i)
        triplet_loss = triplet_loss + temp_loss

    return triplet_loss / max(count, 1)


def hard_triplet_dig(anchor, positive, negative):
    """
    Args:
        anchor: [length, c]
        positive: [nums_x, c]
        negative: [nums_y, c]
    Returns: triplet loss
    """
    edu_distance_pos = F.pairwise_distance(F.normalize(anchor, p=2, dim=-1),
                                                torch.mean(F.normalize(positive, p=2, dim=-1), dim=0, keepdim=True))

    edu_distance_neg = F.pairwise_distance(F.normalize(anchor, p=2, dim=-1), torch.mean(F.normalize(negative, p=2, dim=-1), dim=0, keepdim=True))

    neg_val, _ = edu_distance_neg.sort()
    pos_val, _ = edu_distance_pos.sort()

    triplet_loss = max(0, 0.5 + pos_val[0] - neg_val[0])   # 0.5
    return triplet_loss

class MatchingNet(nn.Module):
    def __init__(self, backbone_name):
        super(MatchingNet, self).__init__()

        self.backbone = resnet.__dict__[backbone_name](pretrained=True)

        self.layer0 = nn.Sequential(self.backbone.conv1, self.backbone.bn1, self.backbone.relu, self.backbone.maxpool)
        self.layer1, self.layer2, self.layer3 = self.backbone.layer1, self.backbone.layer2, self.backbone.layer3

        self.model = clip.load("RN50")[0]

        self.gam = Attention(in_channels=512)

        self.down_query = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3))

        self.down_supp = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3))

        self.down_text_cp = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3))

        self.down_text_up = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.3))

    def forward(self, img_q, class_text, erode, dilate):
        # Text Feature by CLIP
        back_text_1 = clip.tokenize(['background'])
        back_text_1 = torch.stack([back_text_1] * img_q.shape[0], dim=0).squeeze(1).cuda()

        CP = self.down_text_cp(self.model.encode_text(class_text).unsqueeze(-1).unsqueeze(-1).float())
        UP = self.down_text_up(self.model.encode_text(back_text_1).unsqueeze(-1).unsqueeze(-1).float())

        # Image Feature by ResNet
        with torch.no_grad():
            y = self.layer2(self.layer1(self.layer0(img_q)))
        feature_q = self.down_query(self.layer3(y))

        h, w = img_q.shape[-2:]
        bs = feature_q.shape[0]

        # score map optimization
        out_list_q = self.get_out(feature_q, CP, UP)

        out_1 = F.interpolate(out_list_q[0], size=(h, w), mode="bilinear", align_corners=True)
        out_2 = F.interpolate(out_list_q[1], size=(h, w), mode="bilinear", align_corners=True)

        out_ls = [out_2, out_1]

        # Edge Perception Loss
        if self.training:
            tri_loss = get_triple_loss(CP.squeeze(-1).squeeze(-1), feature_q, dilate.unsqueeze(1),erode.unsqueeze(1))
            out_ls.append(tri_loss)

        return out_ls

    def get_out(self, feature_q, CP, UP):
        out_0 = self.similarity_func(feature_q, CP, UP)
        pred_0 = torch.argmax(out_0, dim=1)
        feature_q0 = self.gam(feature_q, pred_0)

        ##################### Initial Score Map Optimization #####################
        SSFP_1, SSBP_1, ASFP_1, ASBP_1 = self.Image_Feature_Selecting(feature_q0, out_0)

        FP_1 = CP * 0.5 + SSFP_1.cuda() * 0.5
        BP_1 = SSBP_1 * 0.3 + ASBP_1 * 0.7

        out_1 = self.similarity_func(feature_q.cuda(), FP_1.cuda(), BP_1.cuda())
        pred_1 = torch.argmax(out_1, dim=1)
        feature_q1 = self.gam(feature_q, pred_1)

        ##################### Final Score Map Optimization #####################
        SSFP_2, SSBP_2, ASFP_2, ASBP_2 = self.Image_Feature_Selecting(feature_q1, out_1)

        BP_2 = SSBP_2 * 0.3 + ASBP_2 * 0.7

        FP_2 = CP * 0.2 + SSFP_1 * 0.3 + SSFP_2 * 0.5
        BP_2 = BP_1 * 0.4 + BP_2 * 0.6

        out_2 = self.similarity_func(feature_q.cuda(), FP_2.cuda(), BP_2.cuda())
        return [out_1, out_2]

    def Image_Feature_Selecting(self, feature_q, out):
        bs = feature_q.shape[0]
        pred_1 = out.softmax(1)
        pred_1 = pred_1.view(bs, 2, -1)
        pred_fg = pred_1[:, 1]
        pred_bg = pred_1[:, 0]
        fg_ls = []
        bg_ls = []
        fg_local_ls = []
        bg_local_ls = []
        for epi in range(bs):
            fg_thres = 0.7 #0.9 #0.6
            bg_thres = 0.6 #0.6
            cur_feat = feature_q[epi].view(512, -1)
            f_h, f_w = feature_q[epi].shape[-2:]
            if (pred_fg[epi] > fg_thres).sum() > 0:
                fg_feat = cur_feat[:, (pred_fg[epi]>fg_thres)] #.mean(-1)
            else:
                fg_feat = cur_feat[:, torch.topk(pred_fg[epi], 12).indices] #.mean(-1)
            if (pred_bg[epi] > bg_thres).sum() > 0:
                bg_feat = cur_feat[:, (pred_bg[epi]>bg_thres)] #.mean(-1)
            else:
                bg_feat = cur_feat[:, torch.topk(pred_bg[epi], 12).indices] #.mean(-1)
            # global proto
            fg_proto = fg_feat.mean(-1)
            bg_proto = bg_feat.mean(-1)
            fg_ls.append(fg_proto.unsqueeze(0))
            bg_ls.append(bg_proto.unsqueeze(0))

            # local proto
            fg_feat_norm = fg_feat / torch.norm(fg_feat, 2, 0, True) # 1024, N1
            bg_feat_norm = bg_feat / torch.norm(bg_feat, 2, 0, True) # 1024, N2
            cur_feat_norm = cur_feat / torch.norm(cur_feat, 2, 0, True) # 1024, N3

            cur_feat_norm_t = cur_feat_norm.t() # N3, 1024
            fg_sim = torch.matmul(cur_feat_norm_t, fg_feat_norm) * 2.0 # N3, N1
            bg_sim = torch.matmul(cur_feat_norm_t, bg_feat_norm) * 2.0 # N3, N2

            fg_sim = fg_sim.softmax(-1)
            bg_sim = bg_sim.softmax(-1)

            fg_proto_local = torch.matmul(fg_sim, fg_feat.t()) # N3, 1024
            bg_proto_local = torch.matmul(bg_sim, bg_feat.t()) # N3, 1024

            fg_proto_local = fg_proto_local.t().view(512, f_h, f_w).unsqueeze(0) # 1024, N3
            bg_proto_local = bg_proto_local.t().view(512, f_h, f_w).unsqueeze(0) # 1024, N3

            fg_local_ls.append(fg_proto_local)
            bg_local_ls.append(bg_proto_local)

        # global proto
        new_fg = torch.cat(fg_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg = torch.cat(bg_ls, 0).unsqueeze(-1).unsqueeze(-1)

        # local proto
        new_fg_local = torch.cat(fg_local_ls, 0).unsqueeze(-1).unsqueeze(-1)
        new_bg_local = torch.cat(bg_local_ls, 0)

        return new_fg, new_bg, new_fg_local, new_bg_local

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1) # 24,1024,50,50    24,1024,1,1     k=2
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)


        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        #24,2,50,50
        return out

    def masked_average_pooling(self, feature, mask):
        mask = F.interpolate(mask.unsqueeze(1), size=feature.shape[-2:], mode='bilinear', align_corners=True)
        masked_feature = torch.sum(feature * mask, dim=(2, 3)) \
                         / (mask.sum(dim=(2, 3)) + 1e-5)
        return masked_feature

    def train_mode(self):
        self.train()
        self.backbone.eval()
        self.model.eval()






