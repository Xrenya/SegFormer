import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegformerEfficientSelfAttention(nn.Module):
    """SegFormer's efficient self-attention mechanism.
    Employs the sequence reduction process introduced in the [PvT
    paper](https://arxiv.org/abs/2102.12122)."""

    def __init__(self, config, hidden_size, num_attention_heads,
                 sequence_reduction_ratio):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads

        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size})"
                " is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})")

        self.attention_head_size = int(self.hidden_size /
                                       self.num_attention_heads)
        self.all_head_size = (
            self.num_attention_heads * self.attention_head_size
        )

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.sr_ratio = sequence_reduction_ratio
        if sequence_reduction_ratio > 1:
            self.sr = nn.Conv2d(hidden_size,
                                hidden_size,
                                kernel_size=sequence_reduction_ratio,
                                stride=sequence_reduction_ratio)
            self.layer_norm = nn.LayerNorm(hidden_size)

    def transpose_for_scores(self, hidden_states):
        new_shape = hidden_states.size()[:-1] + (self.num_attention_heads,
                                                 self.attention_head_size)
        hidden_states = hidden_states.view(new_shape)
        return hidden_states.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        height,
        width,
    ):
        query_layer = self.transpose_for_scores(self.query(hidden_states))

        if self.sr_ratio > 1:
            batch_size, seq_len, num_channels = hidden_states.shape
            # Reshape to (batch_size, num_channels, height, width)
            hidden_states = hidden_states.permute(0, 2, 1).reshape(
                batch_size, num_channels, height, width)
            # Apply sequence reduction
            hidden_states = self.sr(hidden_states)
            # Reshape back to (batch_size, seq_len, num_channels)
            hidden_states = hidden_states.reshape(batch_size, num_channels,
                                                  -1).permute(0, 2, 1)
            hidden_states = self.layer_norm(hidden_states)

        key_layer = self.transpose_for_scores(
            self.key(hidden_states)
        )
        value_layer = self.transpose_for_scores(
            self.value(hidden_states)
        )

        # Take the dot product between "query" and "key"
        # to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer


class SegformerSelfOutput(nn.Module):

    def __init__(self, config, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SegformerAttention(nn.Module):

    def __init__(self, config, hidden_size, num_attention_heads,
                 sequence_reduction_ratio):
        super().__init__()
        self.self = SegformerEfficientSelfAttention(
            config=config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.output = SegformerSelfOutput(config, hidden_size=hidden_size)

    def forward(self, hidden_states, height, width):
        self_outputs = self.self(hidden_states, height, width)
        attention_output = self.output(self_outputs, hidden_states)
        return attention_output


class SegformerOverlapPatchEmbeddings(nn.Module):
    """Construct the overlapping patch embeddings."""

    def __init__(self, patch_size, stride, num_channels, hidden_size):
        super().__init__()
        self.proj = nn.Conv2d(
            num_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=stride,
            padding=patch_size // 2,
        )

        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        embeddings = self.proj(pixel_values)
        _, _, height, width = embeddings.shape
        # (batch_size, num_channels, height, width) ->
        # (batch_size, num_channels, height*width) ->
        # (batch_size, height*width, num_channels)
        # this can be fed to a Transformer layer
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.layer_norm(embeddings)
        return embeddings, height, width


def drop_path(input: torch.Tensor,
              drop_prob: float = 0.0,
              training: bool = False) -> torch.Tensor:
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """  # noqa
    if drop_prob == 0.0 or not training:
        return input
    keep_prob = 1 - drop_prob
    shape = (input.shape[0], ) + (1, ) * (
        input.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(
        shape, dtype=input.dtype, device=input.device)
    random_tensor.floor_()  # binarize
    output = input.div(keep_prob) * random_tensor
    return output


class SegformerDropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""  # noqa

    def __init__(self, drop_prob=None) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return drop_path(hidden_states, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class SegformerLayer(nn.Module):
    """This corresponds to the Block class in the original implementation."""  # noqa

    def __init__(self, config, hidden_size, num_attention_heads, drop_path,
                 sequence_reduction_ratio, mlp_ratio):
        super().__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.attention = SegformerAttention(
            config,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            sequence_reduction_ratio=sequence_reduction_ratio,
        )
        self.drop_path = SegformerDropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        mlp_hidden_size = int(hidden_size * mlp_ratio)
        self.mlp = SegformerMixFFN(config,
                                   in_features=hidden_size,
                                   hidden_features=mlp_hidden_size)

    def forward(self, hidden_states, height, width):
        attention_output = self.attention(
            self.layer_norm_1(
                hidden_states
            ),  # in Segformer, layernorm is applied before self-attention
            height,
            width,
        )

        # first residual connection (with stochastic depth)
        attention_output = self.drop_path(attention_output)
        hidden_states = attention_output + hidden_states

        mlp_output = self.mlp(self.layer_norm_2(hidden_states), height, width)

        # second residual connection (with stochastic depth)
        mlp_output = self.drop_path(mlp_output)
        layer_output = mlp_output + hidden_states

        return layer_output


class SegformerDWConv(nn.Module):

    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, hidden_states, height, width):
        batch_size, seq_len, num_channels = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).view(
            batch_size, num_channels, height, width)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        return hidden_states


class SegformerMixFFN(nn.Module):

    def __init__(self,
                 config,
                 in_features,
                 hidden_features=None,
                 out_features=None):
        super().__init__()
        out_features = out_features or in_features
        self.dense1 = nn.Linear(in_features, hidden_features)
        self.dwconv = SegformerDWConv(hidden_features)
        self.intermediate_act_fn = config.hidden_act
        self.dense2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, height, width):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.dwconv(hidden_states, height, width)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class SegformerEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # stochastic depth decay rule
        drop_path_decays = [
            x.item() for x in torch.linspace(0, config.drop_path_rate,
                                             sum(config.depths))
        ]

        # patch embeddings
        embeddings = []
        for i in range(config.num_encoder_blocks):
            embeddings.append(
                SegformerOverlapPatchEmbeddings(
                    patch_size=config.patch_sizes[i],
                    stride=config.strides[i],
                    num_channels=config.num_channels
                    if i == 0 else config.hidden_sizes[i - 1],
                    hidden_size=config.hidden_sizes[i],
                ))
        self.patch_embeddings = nn.ModuleList(embeddings)

        # Transformer blocks
        blocks = []
        cur = 0
        for i in range(config.num_encoder_blocks):
            # each block consists of layers
            layers = []
            if i != 0:
                cur += config.depths[i - 1]
            for j in range(config.depths[i]):
                layers.append(
                    SegformerLayer(
                        config,
                        hidden_size=config.hidden_sizes[i],
                        num_attention_heads=config.num_attention_heads[i],
                        drop_path=drop_path_decays[cur + j],
                        sequence_reduction_ratio=config.sr_ratios[i],
                        mlp_ratio=config.mlp_ratios[i],
                    ))
            blocks.append(nn.ModuleList(layers))

        self.block = nn.ModuleList(blocks)

        # Layer norms
        self.layer_norm = nn.ModuleList([
            nn.LayerNorm(config.hidden_sizes[i])
            for i in range(config.num_encoder_blocks)
        ])

    def forward(self, pixel_values: torch.FloatTensor):
        batch_size = pixel_values.shape[0]
        all_hidden_states = ()
        hidden_states = pixel_values
        for idx, x in enumerate(
                zip(self.patch_embeddings, self.block, self.layer_norm)):
            embedding_layer, block_layer, norm_layer = x
            # first, obtain patch embeddings
            hidden_states, height, width = embedding_layer(hidden_states)
            # second, send embeddings through blocks
            for i, blk in enumerate(block_layer):
                hidden_states = blk(hidden_states, height, width)
            # third, apply layer norm
            hidden_states = norm_layer(hidden_states)
            # fourth, optionally reshape back to
            # (batch_size, num_channels, height, width)
            if idx != len(self.patch_embeddings) - 1 or (
                    idx == len(self.patch_embeddings) - 1
                    and self.config.reshape_last_stage):
                hidden_states = hidden_states.reshape(batch_size, height,
                                                      width, -1).permute(
                                                          0, 3, 1,
                                                          2).contiguous()
            all_hidden_states = all_hidden_states + (hidden_states, )
        return all_hidden_states


class SegformerModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        # hierarchical Transformer encoder
        self.encoder = SegformerEncoder(config)

    def forward(self, pixel_values: torch.FloatTensor):
        encoder_outputs = self.encoder(pixel_values, )

        return encoder_outputs


class SegformerMLP(nn.Module):
    """
    Linear Embedding.
    """

    def __init__(self, config, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, config.decoder_hidden_size)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class SegformerDecodeHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        # linear layers which will unify the channel dimension of
        # each of the encoder blocks to the same config.decoder_hidden_size  # noqa 
        mlps = []
        for i in range(config.num_encoder_blocks):
            mlp = SegformerMLP(config, input_dim=config.hidden_sizes[i])
            mlps.append(mlp)
        self.linear_c = nn.ModuleList(mlps)

        # the following 3 layers implement the ConvModule
        # of the original implementation
        self.linear_fuse = nn.Conv2d(
            in_channels=config.decoder_hidden_size * config.num_encoder_blocks,
            out_channels=config.decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )
        self.batch_norm = nn.BatchNorm2d(config.decoder_hidden_size)
        self.activation = nn.ReLU()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Conv2d(config.decoder_hidden_size,
                                    config.num_labels,
                                    kernel_size=1)

        self.config = config
        self.image_size = (config.image_size, config.image_size)

    def forward(self,
                encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, mlp in zip(encoder_hidden_states,
                                             self.linear_c):
            if (
                self.config.reshape_last_stage is False
                and encoder_hidden_state.ndim == 3
            ):
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (encoder_hidden_state.reshape(
                    batch_size, height, width, -1).permute(0, 3, 1,
                                                           2).contiguous())

            # unify channel dimension
            height, width = encoder_hidden_state.shape[
                2], encoder_hidden_state.shape[3]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(
                batch_size, -1, height, width)
            # upsample
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state,
                size=encoder_hidden_states[0].size()[2:],
                mode="bilinear",
                align_corners=False)
            all_hidden_states += (encoder_hidden_state, )

        hidden_states = self.linear_fuse(
            torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # logits are of shape (batch_size, num_labels, height/4, width/4)
        logits = self.classifier(hidden_states)

        logits = F.interpolate(logits,
                               size=self.image_size,
                               mode="bilinear",
                               align_corners=False)

        return logits


class SegformerForSemanticSegmentation(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.segformer = SegformerModel(config)
        self.decode_head = SegformerDecodeHead(config)

    def forward(self, pixel_values: torch.FloatTensor):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth semantic segmentation maps for computing the loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels > 1`, a classification loss is computed (Cross-Entropy).

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
        >>> from PIL import Image
        >>> import requests

        >>> image_processor = AutoImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
        >>> model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> logits = outputs.logits  # shape (batch_size, num_labels, height/4, width/4)
        >>> list(logits.shape)
        [1, 150, 128, 128]
        ```"""  # noqa
        outputs = self.segformer(pixel_values)

        logits = self.decode_head(outputs)

        return logits


class SegformerConfig:
    r"""
    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        num_encoder_blocks (`int`, *optional*, defaults to 4):
            The number of encoder blocks (i.e. stages in the Mix Transformer encoder).
        depths (`List[int]`, *optional*, defaults to `[2, 2, 2, 2]`):
            The number of layers in each encoder block.
        sr_ratios (`List[int]`, *optional*, defaults to `[8, 4, 2, 1]`):
            Sequence reduction ratios in each encoder block.
        hidden_sizes (`List[int]`, *optional*, defaults to `[32, 64, 160, 256]`):
            Dimension of each of the encoder blocks.
        patch_sizes (`List[int]`, *optional*, defaults to `[7, 3, 3, 3]`):
            Patch size before each encoder block.
        strides (`List[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
            Stride before each encoder block.
        num_attention_heads (`List[int]`, *optional*, defaults to `[1, 2, 5, 8]`):
            Number of attention heads for each attention layer in each block of the Transformer encoder.
        mlp_ratios (`List[int]`, *optional*, defaults to `[4, 4, 4, 4]`):
            Ratio of the size of the hidden layer compared to the size of the input layer of the Mix FFNs in the
            encoder blocks.
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        classifier_dropout_prob (`float`, *optional*, defaults to 0.1):
            The dropout probability before the classification head.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            The dropout probability for stochastic depth, used in the blocks of the Transformer encoder.
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the layer normalization layers.
        decoder_hidden_size (`int`, *optional*, defaults to 256):
            The dimension of the all-MLP decode head.
        semantic_loss_ignore_index (`int`, *optional*, defaults to 255):
            The index that is ignored by the loss function of the semantic segmentation model.

    Example:

    ```python
    >>> from transformers import SegformerModel, SegformerConfig

    >>> # Initializing a SegFormer nvidia/segformer-b0-finetuned-ade-512-512 style configuration
    >>> configuration = SegformerConfig()

    >>> # Initializing a model from the nvidia/segformer-b0-finetuned-ade-512-512 style configuration
    >>> model = SegformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""  # noqa

    model_type = "segformer"

    def __init__(
        self,
        num_channels=3,
        num_encoder_blocks=4,
        depths=[2, 2, 2, 2],
        sr_ratios=[8, 4, 2, 1],
        hidden_sizes=[32, 64, 160, 256],
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
        num_attention_heads=[1, 2, 5, 8],
        mlp_ratios=[4, 4, 4, 4],
        hidden_act=nn.GELU(),
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout_prob=0.1,
        initializer_range=0.02,
        drop_path_rate=0.1,
        layer_norm_eps=1e-6,
        decoder_hidden_size=256,
        semantic_loss_ignore_index=255,
        num_labels=150,
        image_size=512,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.depths = depths
        self.sr_ratios = sr_ratios
        self.hidden_sizes = hidden_sizes
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.mlp_ratios = mlp_ratios
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.initializer_range = initializer_range
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.decoder_hidden_size = decoder_hidden_size
        self.reshape_last_stage = kwargs.get("reshape_last_stage", True)
        self.semantic_loss_ignore_index = semantic_loss_ignore_index
        self.num_labels = num_labels
        self.image_size = image_size
