from collections import OrderedDict
import torch
import torch.nn.functional as F
from torch import nn
from transformers import ViTModel
from transformers import ViTForImageClassification


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=0.1)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(
        self,
        input_resolution: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        num_classes: int,
        output_dim: int=-1
    ):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.transformer = Transformer(width, layers, heads)
        self.ln_post = LayerNorm(width)

        # If teacher's output dim is different, we want to project
        # teacher's embeddings into student's width.
        output_dim = width if output_dim == -1 else output_dim
        #self.proj =  nn.Parameter(scale * torch.randn(width, width))
        self.classifier = nn.Linear(width, num_classes)
        
        # Create a projection layer for the cluster if needed
        # cluster dim should be same as width
        self.cluster_projection = nn.Linear(output_dim, width)

    def forward(self, x: torch.Tensor, cluster: torch.Tensor=None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        hard_labels = None
        if self.classifier is not None:
            hard_labels = F.softmax(self.classifier(x), dim=-1)

        projected_cluster = None
        if cluster is not None:
            projected_cluster = self.cluster_projection(cluster)

        return x, hard_labels, projected_cluster


class TeacherViT(torch.nn.Module):
    def __init__(
        self,
        model_name
    ):
        super().__init__()
        self.model = ViTModel.from_pretrained(model_name)

    def forward(self, x):
        try:
            outputs = self.model(x)
        except:
            outputs = self.model(**x)
        last_hidden_states = outputs.last_hidden_state
        # Get embeddings from the [CLS] token
        outputs = last_hidden_states[:, 0, :]
        return outputs

class BaselineViT(torch.nn.Module):
    def __init__(
        self,
        model_name
    ):
        super().__init__()
        self.model = ViTForImageClassification.from_pretrained(model_name)

    def forward(self, x):
        with torch.no_grad():
            outputs = self.model(x)

        return F.softmax(outputs.logits, dim=-1)
