import torch.nn as nn
import torch
from block import ConvBlock
from block import Gconv
import torch.nn.functional as F
from block import ResBlock
import math


class Discriminator(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, spec_norm=True, LR=0.2):
        super(Discriminator, self).__init__()
        self.main = list()
        self.main.append(ConvBlock(4, 16, spec_norm, stride=2, LR=LR))  # 256 -> 128
        self.main.append(ConvBlock(16, 32, spec_norm, stride=2, LR=LR))  # 128 -> 64
        self.main.append(ConvBlock(32, 64, spec_norm, stride=2, LR=LR))  # 64 -> 32
        self.main.append(ConvBlock(64, 128, spec_norm, stride=2, LR=LR))  # 32 -> 16
        self.main.append(nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)


class ResBlockNet(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlockNet, self).__init__()
        self.main = list()
        self.main.append(ResBlock(in_channels, out_channels))
        self.main.append(ResBlock(out_channels, out_channels))
        self.main.append(ResBlock(out_channels, out_channels))
        self.main.append(ResBlock(out_channels, out_channels))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x) + x


class Encoder(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, in_channels=3, spec_norm=False, LR=0.2):
        super(Encoder, self).__init__()

        self.layer1 = ConvBlock(in_channels, 16, spec_norm, LR=LR)  # 256
        self.layer2 = ConvBlock(16, 16, spec_norm, LR=LR)  # 256
        self.layer3 = ConvBlock(16, 32, spec_norm, stride=2, LR=LR)  # 128
        self.layer4 = ConvBlock(32, 32, spec_norm, LR=LR)  # 128
        self.layer5 = ConvBlock(32, 64, spec_norm, stride=2, LR=LR)  # 64
        self.layer6 = ConvBlock(64, 64, spec_norm, LR=LR)  # 64
        self.layer7 = ConvBlock(64, 128, spec_norm, stride=2, LR=LR)  # 32
        self.layer8 = ConvBlock(128, 128, spec_norm, LR=LR)  # 32
        self.layer9 = ConvBlock(128, 256, spec_norm, stride=2, LR=LR)  # 16
        self.layer10 = ConvBlock(256, 256, spec_norm, LR=LR)  # 16
        self.down_sampling = nn.AdaptiveAvgPool2d((16, 16))

    def forward(self, x):
        feature_map1 = self.layer1(x)  # torch.Size([b, 16, 256, 256])
        feature_map2 = self.layer2(feature_map1)  # torch.Size([b, 16, 256, 256])
        feature_map3 = self.layer3(feature_map2)  # torch.Size([b, 32, 128, 128])
        feature_map4 = self.layer4(feature_map3)  # torch.Size([b, 32, 128, 128])
        feature_map5 = self.layer5(feature_map4)  # torch.Size([b, 64, 64, 64])
        feature_map6 = self.layer6(feature_map5)  # torch.Size([b, 64, 64, 64])
        feature_map7 = self.layer7(feature_map6)  # torch.Size([b, 128, 32, 32])
        feature_map8 = self.layer8(feature_map7)  # torch.Size([b, 128, 32, 32])
        feature_map9 = self.layer9(feature_map8)  # torch.Size([b, 256, 16, 16])
        feature_map10 = self.layer10(feature_map9)  # torch.Size([b, 256, 16, 16])

        down_feature_map1 = self.down_sampling(feature_map1)
        down_feature_map2 = self.down_sampling(feature_map2)
        down_feature_map3 = self.down_sampling(feature_map3)
        down_feature_map4 = self.down_sampling(feature_map4)
        down_feature_map5 = self.down_sampling(feature_map5)
        down_feature_map6 = self.down_sampling(feature_map6)
        down_feature_map7 = self.down_sampling(feature_map7)
        down_feature_map8 = self.down_sampling(feature_map8)

        output = torch.cat([down_feature_map1,
                            down_feature_map2,
                            down_feature_map3,
                            down_feature_map4,
                            down_feature_map5,
                            down_feature_map6,
                            down_feature_map7,
                            down_feature_map8,
                            feature_map9,
                            feature_map10,
                            ], dim=1)
        # torch.Size([2, 992, 16, 16])
        feature_list = [feature_map1,
                        feature_map2,
                        feature_map3,
                        feature_map4,
                        feature_map5,
                        feature_map6,
                        feature_map7,
                        feature_map8,
                        feature_map9,
                        feature_map10,
                        ]

        return output, feature_list


class Decoder(nn.Module):

    def __init__(self, spec_norm=False, LR=0.2):
        super(Decoder, self).__init__()
        self.layer10 = ConvBlock(992 + 992, 256, spec_norm, LR=LR)  # 16->16
        # self.layer10 = ConvBlock(992 + 992, 256, spec_norm, LR=LR)  # 16->16
        self.layer9 = ConvBlock(256 + 256, 256, spec_norm, LR=LR)  # 16->16
        self.layer8 = ConvBlock(256 + 128, 128, spec_norm, LR=LR, up=True)  # 16->32
        self.layer7 = ConvBlock(128 + 128, 128, spec_norm, LR=LR)  # 32->32
        self.layer6 = ConvBlock(128 + 64, 64, spec_norm, LR=LR, up=True)  # 32-> 64
        self.layer5 = ConvBlock(64 + 64, 64, spec_norm, LR=LR)  # 64 -> 64
        self.layer4 = ConvBlock(64 + 32, 32, spec_norm, LR=LR, up=True)  # 64 -> 128
        self.layer3 = ConvBlock(32 + 32, 32, spec_norm, LR=LR)  # 128 -> 128
        self.layer2 = ConvBlock(32 + 16, 16, spec_norm, LR=LR, up=True)  # 128 -> 256
        self.layer1 = ConvBlock(16 + 16, 16, spec_norm, LR=LR)  # 256 -> 256
        self.last_conv = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x, feature_list):
        feature_map10 = self.layer10(x)
        # feature_map10 = self.layer10(torch.cat([x, feature_list[-1]], dim=1))
        feature_map9 = self.layer9(torch.cat([feature_map10, feature_list[-2]], dim=1))
        feature_map8 = self.layer8(feature_map9, feature_list[-3])
        feature_map7 = self.layer7(torch.cat([feature_map8, feature_list[-4]], dim=1))
        feature_map6 = self.layer6(feature_map7, feature_list[-5])
        feature_map5 = self.layer5(torch.cat([feature_map6, feature_list[-6]], dim=1))
        feature_map4 = self.layer4(feature_map5, feature_list[-7])
        feature_map3 = self.layer3(torch.cat([feature_map4, feature_list[-8]], dim=1))
        feature_map2 = self.layer2(feature_map3, feature_list[-9])
        feature_map1 = self.layer1(torch.cat([feature_map2, feature_list[-10]], dim=1))
        feature_map0 = self.last_conv(feature_map1)

        return self.tanh(feature_map0)


class SCFT_Module(nn.Module):
    def __init__(self, channels=992, margin=12):
        super(SCFT_Module, self).__init__()
        self.scaling_factor = math.sqrt(channels)
        self.channels = channels
        self.softmax = nn.Softmax(dim=-1)
        self.margin = margin

        self.w_q = nn.Linear(channels, channels)
        self.w_k = nn.Linear(channels, channels)
        self.w_v = nn.Linear(channels, channels)

    def forward(self, v_s, v_r):
        bs, ch, h, w = v_s.size()
        v_s = v_s.view(bs, ch, h * w).permute(0, 2, 1)
        v_r = v_r.view(bs, ch, h * w).permute(0, 2, 1)

        q_result = self.w_q(v_s)
        k_result = self.w_k(v_r)
        v_result = self.w_v(v_r)
        # torch.Size([b, 256, 992])

        dot = torch.einsum('bik,bjk->bij', q_result, k_result)  # torch.Size([b, skt_wh, ref_wh])
        attention_map = self.softmax(dot / self.scaling_factor)

        v_star = torch.bmm(attention_map, v_result)  # torch.Size([b, 256, 992])

        v_sum = (v_star + v_s).permute(0, 2, 1)  # torch.Size([b, 992, 256])
        v_sum = v_sum.view(bs, ch, h, w)  # torch.Size([b, 992, 16, 16])

        return v_sum

    def predict(self, v_s, v_r):
        bs, ch, h, w = v_s.size()
        v_s = v_s.view(bs, ch, h * w).permute(0, 2, 1)
        v_r = v_r.view(bs, ch, h * w).permute(0, 2, 1)

        q_result = self.w_q(v_s)
        k_result = self.w_k(v_r)
        v_result = self.w_v(v_r)
        # torch.Size([b, 256, 992])

        dot = torch.einsum('bik,bjk->bij', q_result, k_result)  # torch.Size([b, skt_wh, ref_wh])
        attention_map = self.softmax(dot / self.scaling_factor)

        v_star = torch.bmm(attention_map, v_result)  # torch.Size([b, 256, 992])

        v_sum = (v_star + v_s).permute(0, 2, 1)  # torch.Size([b, 992, 256])
        v_sum = v_sum.view(bs, ch, h, w)  # torch.Size([b, 992, 16, 16])

        return v_sum, v_s

    def gradient_cosine(self, v_s, v_r):
        Z_grad = list()
        Q_grad = list()
        K_grad = list()
        V_grad = list()

        def grad_hook_Z(grad):
            Z_grad.append(grad)

        def grad_hook_Q(grad):
            Q_grad.append(grad)

        def grad_hook_K(grad):
            K_grad.append(grad)

        def grad_hook_V(grad):
            V_grad.append(grad)

        bs, ch, h, w = v_s.size()
        v_s = v_s.view(bs, ch, h * w).permute(0, 2, 1)
        v_r = v_r.view(bs, ch, h * w).permute(0, 2, 1)
        v_s_q = v_s.clone()
        v_s_z = v_s.clone()
        v_r_k = v_r.clone()
        v_r_v = v_r.clone()

        q_result = self.w_q(v_s_q)
        k_result = self.w_k(v_r_k)
        v_result = self.w_v(v_r_v)
        # torch.Size([b, 256, 992])

        dot = torch.einsum('bik,bjk->bij', q_result, k_result)  # torch.Size([b, skt_wh, ref_wh])
        attention_map = self.softmax(dot / self.scaling_factor)

        v_star = torch.bmm(attention_map, v_result)  # torch.Size([b, 256, 992])

        v_sum = (v_star + v_s_z).permute(0, 2, 1)  # torch.Size([b, 992, 256])
        v_sum = v_sum.view(bs, ch, h, w)  # torch.Size([b, 992, 16, 16])

        hz = v_s_z.register_hook(grad_hook_Z)
        hq = v_s_q.register_hook(grad_hook_Q)
        hk = v_r_k.register_hook(grad_hook_K)
        hv = v_r_v.register_hook(grad_hook_V)

        return v_sum, Z_grad, Q_grad, K_grad, V_grad


class GNN(nn.Module):
    def __init__(self, channel):
        super(GNN, self).__init__()
        self.channel = channel
        self.gcn3 = Gconv(in_features=channel, out_features=channel)
        self.gcn4 = Gconv(in_features=channel, out_features=channel)

    @staticmethod
    def build_graph(src, tgt):
        """
        src -> (b,wh,c)
        tgt -> (b,wh,c)
        """
        with torch.no_grad():
            graph = src.bmm(tgt.permute(0, 2, 1))
            graph = F.softmax(graph, dim=-1)
            graph = F.normalize(graph, p=1, dim=-2)

        return graph

    @staticmethod
    def build_attention(src, tgt):
        """
        src -> (b,wh,c)
        tgt -> (b,wh,c)
        """
        with torch.no_grad():
            # norm_src = F.normalize(src, dim=-1)
            # norm_tgt = F.normalize(tgt, dim=-1)
            graph = src.bmm(tgt.permute(0, 2, 1))
            graph = F.softmax(graph, dim=-1)
            # graph = norm_src.bmm(norm_tgt.permute(0, 2, 1))
            # graph = F.softmax(graph, dim=-1)
        return graph

    def forward(self, skt, ref):
        b, c, h, w = skt.size()
        skt = skt.view(b, c, h * w).permute(0, 2, 1)
        ref = ref.view(b, c, h * w).permute(0, 2, 1)
        sr = self.build_graph(src=skt, tgt=ref)

        gen = self.gcn3(A=sr, source=skt, message=ref) + skt

        gg = self.build_graph(src=gen, tgt=gen)

        ggen = self.gcn4(A=gg, source=gen, message=gen) + gen

        return ggen

    def predict(self, skt, ref):
        b, c, h, w = skt.size()
        skt = skt.view(b, c, h * w).permute(0, 2, 1)
        ref = ref.view(b, c, h * w).permute(0, 2, 1)
        sr = self.build_graph(src=skt, tgt=ref)

        gen = self.gcn3(A=sr, source=skt, message=ref) + skt

        gg = self.build_graph(src=gen, tgt=gen)

        ggen = self.gcn4(A=gg, source=gen, message=gen) + gen

        return ggen, skt, ggen

    def att(self, skt, ref):
        b, c, h, w = skt.size()
        skt = skt.view(b, c, h * w).permute(0, 2, 1)
        ref = ref.view(b, c, h * w).permute(0, 2, 1)
        sr = self.build_graph(src=skt, tgt=ref)

        gen = self.gcn3(A=sr, source=skt, message=ref) + skt

        gg = self.build_graph(src=gen, tgt=gen)

        ggen = self.gcn4(A=gg, source=gen, message=gen) + gen

        return ggen, self.build_attention(src=skt, tgt=ref)


class Generator(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, spec_norm=False, LR=0.2):
        super(Generator, self).__init__()
        self.encoder_reference = Encoder(in_channels=3, spec_norm=spec_norm, LR=LR)
        self.encoder_sketch = Encoder(in_channels=1, spec_norm=spec_norm, LR=LR)
        self.gnn = GNN(channel=992)
        self.decoder = Decoder()
        self.res_model = ResBlockNet(992, 992)

    def forward(self, reference, sketch):
        # feature extract
        v_r, _ = self.encoder_reference(reference)
        v_s, feature_list = self.encoder_sketch(sketch)
        b, c, h, w = v_r.size()

        # GNN
        v_c = self.gnn(skt=v_s, ref=v_r).view(b, h, w, c).permute(0, 3, 1, 2)
        rv_c = self.res_model(v_c)
        concat = torch.cat([rv_c, v_c], dim=1)

        # decoder
        image = self.decoder(concat, feature_list)
        # image = self.decoder(v_c, feature_list)
        return image

    def predict(self, reference, sketch):
        # feature extract
        v_r, _ = self.encoder_reference(reference)
        v_s, feature_list = self.encoder_sketch(sketch)
        b, c, h, w = v_r.size()

        # GNN
        v_c, before, after = self.gnn.predict(skt=v_s, ref=v_r)
        v_c = v_c.view(b, h, w, c).permute(0, 3, 1, 2)
        rv_c = self.res_model(v_c)
        concat = torch.cat([rv_c, v_c], dim=1)

        # decoder
        image = self.decoder(concat, feature_list)
        # image = self.decoder(v_c, feature_list)
        return image, before, after
