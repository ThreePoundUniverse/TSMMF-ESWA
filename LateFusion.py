import torch
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from einops import rearrange
from torch.autograd import Function
from DualConvLayer import SpatialConvLayer, TemporalConvLayer


class PositionalEmbedding(nn.Module):
    def __init__(self, channels, emb_size, device):
        super().__init__()
        self.channels = channels + 1
        self.pos_emb = nn.Parameter(torch.randn(size=(1, self.channels, emb_size), dtype=torch.float32, device=device),
                                    requires_grad=True)

    def forward(self, x):
        x = self.pos_emb + x
        return x


class ModalityTypeEmbedding(nn.Module):
    def __init__(self, emb_size, token_type_idx=1):
        super().__init__()
        self.token_type_idx = token_type_idx
        self.type_embedding_layer = nn.Embedding(2, emb_size)

    def forward(self, x, mask):
        # x.shape = B, 1 + eeg_tokens + 1 + nirs_tokens, emb_size
        # mask.shape = [1 + eeg_tokens, 1 + nirs_tokens]
        b, _, emb_size = x.shape
        modality_type_emb = torch.ones(b, mask[0] + mask[1], dtype=torch.long, device=x.device)
        modality_type_emb[:, mask[0]::] = 0
        type_emb = self.type_embedding_layer(modality_type_emb)
        x = x + type_emb
        return x


class AddClsToken(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size), requires_grad=True)

    def forward(self, x):
        self.cls_token = self.cls_token.to(x.device)
        x = torch.cat([self.cls_token.repeat(x.shape[0], 1, 1), x], dim=1)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, emb_size, num_heads, dropout, bias=False):
        super().__init__()
        self.emb_size = emb_size
        self.proj_dim = self.emb_size
        self.num_heads = num_heads
        self.queries = nn.Linear(query_size, self.proj_dim, bias=bias)
        self.keys = nn.Linear(key_size, self.proj_dim, bias=bias)
        self.values = nn.Linear(value_size, self.proj_dim, bias=bias)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(self.proj_dim, emb_size, bias=bias),
            nn.Dropout(dropout)
        )
        self.attention_weights = None

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(query), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(key), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(value), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        self.attention_weights = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(self.attention_weights)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, query_size, key_size, value_size, emb_size, num_heads, dropout, bias=False):
        super().__init__()
        self.proj_dim = emb_size
        self.num_heads = num_heads
        self.queries = nn.Linear(query_size, self.proj_dim, bias=bias)
        self.keys = nn.Linear(key_size, self.proj_dim, bias=bias)
        self.values = nn.Linear(value_size, self.proj_dim, bias=bias)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Linear(self.proj_dim, emb_size, bias=bias),
            nn.Dropout(dropout)
        )
        self.attention_weights = None
        self.scaling = emb_size ** (1 / 2)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(query), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(key), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(value), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        self.attention_weights = F.softmax(energy / self.scaling, dim=-1)
        att = self.att_drop(self.attention_weights)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Module):
    def __init__(self, emb_size, expansion=2, dropout=0.5):
        super().__init__()
        self.feed_forward = nn.Sequential(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * emb_size, emb_size),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.feed_forward(x)

        return x


class SelfEncoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, emb_size, num_heads=4, forward_expansion=2, dropout=0.5):
        super(SelfEncoderBlock, self).__init__()

        self.attention = MultiHeadSelfAttention(query_size, key_size, value_size, emb_size, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(emb_size, expansion=forward_expansion, dropout=dropout)

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x):
        # x是主导模态
        res = x
        x = self.norm1(x)
        y = self.attention(x, x, x)
        y = y + res

        res = y
        y2 = self.norm2(y)
        y2 = self.feed_forward(y2)
        y2 = y2 + res

        return y2


class CrossEncoderBlock(nn.Module):
    def __init__(self, query_size, key_size, value_size, emb_size, num_heads=4, forward_expansion=2, dropout=0.5):
        super(CrossEncoderBlock, self).__init__()

        self.attention = MultiHeadAttention(query_size, key_size, value_size, emb_size, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(emb_size, expansion=forward_expansion, dropout=dropout)

        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.norm3 = nn.LayerNorm(emb_size)

    def forward(self, x, y):
        # x是主导模态
        res = x
        x, y = self.norm1(x), self.norm2(y)
        y1 = self.attention(x, y, y)
        y1 = y1 + res

        res = y1
        y2 = self.norm3(y1)
        y2 = self.feed_forward(y2)
        y2 = y2 + res

        return y2


class TransformerCrossEncoder(nn.Module):
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, dropout,
                 device):
        super(TransformerCrossEncoder, self).__init__()
        # self.positional_embedding1 = PositionalEmbedding(channels[0] + 1, emb_size, device=device)
        # self.positional_embedding2 = PositionalEmbedding(channels[1] + 1, emb_size, device=device)

        self.blks = nn.Sequential()
        self.attention_weights = [None] * depth

        for i in range(depth):
            self.blks.add_module("block" + str(i),
                                 CrossEncoderBlock(query_size, key_size, value_size, emb_size, num_heads, expansion,
                                                   dropout))

    def forward(self, x, y):
        # x = self.positional_embedding1(x)
        # y = self.positional_embedding2(y)
        for i, blk in enumerate(self.blks):
            x = blk(x, y)
            self.attention_weights[i] = blk.attention.attention_weights
        return x

    @property
    def cross_attention_weights(self):
        return self.attention_weights


class TransformerCatEncoder(nn.Module):
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, dropout,
                 device):
        super(TransformerCatEncoder, self).__init__()
        self.modality_embedding = ModalityTypeEmbedding(emb_size)
        self.blks = nn.Sequential()
        self.attention_weights = [None] * depth

        for i in range(depth):
            self.blks.add_module("block" + str(i),
                                 SelfEncoderBlock(query_size, key_size, value_size, emb_size, num_heads, expansion,
                                                  dropout))

    def forward(self, x, y, mask=None):
        context = torch.cat([x, y], dim=1)
        context = self.modality_embedding(context, mask)
        for i, blk in enumerate(self.blks):
            context = blk(context)
            self.attention_weights[i] = blk.attention.attention_weights
        return context

    @property
    def self_attention_weights(self):
        return self.attention_weights


class TransformerSelfEncoder(nn.Module):
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, dropout,
                 device):
        super(TransformerSelfEncoder, self).__init__()
        self.blks = nn.Sequential()
        self.attention_weights = [None] * depth
        self.add_token = AddClsToken(emb_size)
        self.positional_embedding = PositionalEmbedding(channels, emb_size, device)
        for i in range(depth):
            self.blks.add_module("block" + str(i),
                                 SelfEncoderBlock(query_size, key_size, value_size, emb_size, num_heads, expansion,
                                                  dropout))

    def forward(self, x, mask=None):
        x = self.add_token(x)
        x = self.positional_embedding(x)
        for i, blk in enumerate(self.blks):
            x = blk(x)
            self.attention_weights[i] = blk.attention.attention_weights
        return x

    @property
    def self_attention_weights(self):
        return self.attention_weights


class Transformer(nn.Module):
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, channels, expansion, device,
                 self_dropout, cross_dropout):
        super().__init__()
        self.eeg_nirs_temporal_spatial_attention_weights = None
        self.eeg_temporal_spatial_attention_weights = None
        self.eeg_temporal_attention_weights = None
        self.eeg_spatial_attention_weights = None
        self.nirs_spatial_attention_weights = None
        self.temporal_mask = [channels[0] + 1, channels[2] + 1]
        self.spatial_mask = [channels[1] + 1, channels[3] + 1]


        self.eeg_spatial_encoder = TransformerSelfEncoder(depth, query_size, key_size, value_size, emb_size, num_heads,
                                                          channels[1], expansion, self_dropout, device)

        self.nirs_spatial_encoder = TransformerSelfEncoder(depth, query_size, key_size, value_size, emb_size,
                                                           num_heads, channels[3], expansion, cross_dropout, device)

        self.eeg_temporal_encoder = TransformerSelfEncoder(depth, query_size, key_size, value_size, emb_size,
                                                           num_heads, channels[0], expansion, cross_dropout, device)

        self.nirs_temporal_encoder = TransformerSelfEncoder(depth, query_size, key_size, value_size, emb_size,
                                                            num_heads, channels[2], expansion, cross_dropout, device)

    def forward(self, temporal_eeg, temporal_nirs, spatial_eeg, spatial_nirs):

        eeg_spatial_outputs = self.eeg_spatial_encoder(spatial_eeg)
        nirs_spatial_outputs = self.nirs_spatial_encoder(spatial_nirs)

        eeg_temporal_outputs = self.eeg_temporal_encoder(temporal_eeg)
        nirs_temporal_outputs = self.nirs_temporal_encoder(temporal_nirs)

        return [eeg_temporal_outputs[:, 0], nirs_temporal_outputs[:, 0],
                eeg_spatial_outputs[:, 0], nirs_spatial_outputs[:, 0]]

    @property
    def get_eeg_attention_weights(self):
        return [self.eeg_temporal_attention_weights, self.eeg_spatial_attention_weights]

    @property
    def get_nirs_spatial_attention_weights(self):
        return [self.nirs_spatial_attention_weights, ]

    @property
    def get_cross_attention_weights(self):
        return [self.eeg_nirs_temporal_spatial_attention_weights, self.eeg_temporal_spatial_attention_weights]


class AttentionFusion(nn.Module):
    def __init__(self, emb_size):
        super(AttentionFusion, self).__init__()
        self.weight = nn.Parameter(torch.randn(emb_size, 1), requires_grad=True)
        self.softmax = nn.Softmax(-1)
        self.alpha = None

    def forward(self, out):
        o = torch.cat([i @ self.weight for i in out], dim=-1)
        self.alpha = self.softmax(o)
        outputs = sum([i * self.alpha[:, index].unsqueeze(1) for index, i in enumerate(out)])
        return outputs


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class ClassificationHead(nn.Module):
    def __init__(self, num_classes, num_domains, emb_size, domain_generalization, dropout):
        super(ClassificationHead, self).__init__()
        self.dropout = dropout
        self.fusion = nn.Sequential(
            nn.Linear(emb_size * 4, emb_size * 2),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(emb_size * 2, num_classes)
        )

    def forward(self, out):
        eeg_temporal, nirs_temporal, eeg_spatial, nirs_spatial = out

        out = self.fusion(torch.cat([eeg_temporal, nirs_temporal, eeg_spatial, nirs_spatial], dim=-1))

        return out


class HybridTransformer(nn.Module):
    def __init__(self, depth, query_size, key_size, value_size, emb_size, num_heads, expansion, domain_generalization,
                 conv_dropout, self_dropout, cross_dropout, cls_dropout, num_classes, num_domains, device):
        super().__init__()
        self.spatial_conv_layer = SpatialConvLayer(emb_size, conv_dropout)
        self.temporal_conv_layer = TemporalConvLayer(emb_size, conv_dropout)

        with torch.no_grad():
            eeg, nirs = torch.randn(1, 60, 1000), torch.randn(1, 40, 40)
            eeg_temporal_token, nirs_temporal_token = self.temporal_conv_layer(eeg, nirs)
            eeg_spatial_token, nirs_spatial_token = self.spatial_conv_layer(eeg, nirs)
            channels = [eeg_temporal_token.shape[-1], eeg_spatial_token.shape[-2], nirs_temporal_token.shape[-1], nirs_spatial_token.shape[-2]]

        self.transformer = Transformer(depth, query_size, key_size, value_size, emb_size, num_heads, channels,
                                       expansion, device, self_dropout, cross_dropout)
        self.classify = ClassificationHead(num_classes, num_domains, emb_size, domain_generalization, cls_dropout)

    def forward(self, eeg, nirs):
        spatial_eeg, spatial_nirs = self.spatial_conv_layer(eeg, nirs)
        temporal_eeg, temporal_nirs = self.temporal_conv_layer(eeg, nirs)
        temporal_eeg, temporal_nirs = temporal_eeg.squeeze(-2).permute(0, 2, 1), temporal_nirs.squeeze(-2).permute(0, 2, 1)
        spatial_eeg, spatial_nirs = spatial_eeg.squeeze(-1).permute(0, 2, 1), spatial_nirs.squeeze(-1).permute(0, 2, 1)
        out1 = self.transformer(temporal_eeg, temporal_nirs, spatial_eeg, spatial_nirs)
        outputs = self.classify(out1)

        return outputs
