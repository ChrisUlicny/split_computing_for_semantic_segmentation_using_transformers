import torch
from torch import nn, einsum
import numpy as np
import torch.nn.functional as f
from einops import rearrange
from utils.ModelUtils import SplitUnit


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PostNorm(nn.Module):
    # dim here is basically the number of channels
    # each token is normalized with respect to itself
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        # return self.fn(self.norm(x), **kwargs)
        # Post norm like in version 2 swin
        return self.norm(self.fn(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class CyclicShift(nn.Module):
    def __init__(self, displacement):
        super().__init__()
        self.displacement = displacement

    def forward(self, x):
        return torch.roll(x, shifts=(self.displacement, self.displacement), dims=(1, 2))


def create_mask(window_size, displacement, upper_lower, left_right):
    mask = torch.zeros(window_size ** 2, window_size ** 2)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return mask


def get_relative_distances(window_size):
    indices = torch.tensor(np.array([[x, y] for x in range(window_size) for y in range(window_size)]))
    distances = indices[None, :, :] - indices[:, None, :]
    return distances



class WindowAttention(nn.Module):
    def __init__(self, dim, heads, head_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        # equal to the number of channels
        inner_dim = head_dim * heads

        self.heads = heads
        # 0.5 because we are doing square root -> scaling dot product inside softmax
        self.scale = head_dim ** -0.5
        # window size is 7
        self.window_size = window_size
        self.relative_pos_embedding = relative_pos_embedding
        self.shifted = shifted

        # tau for cosine similarity - requires_grad is true because it is learnable parameter
        self.tau = nn.Parameter(torch.tensor(0.01), requires_grad=True)

        # when we do shift the regions that shifted outside picture are going to be empty, so we have to pad them
        if self.shifted:
            # how much are we shifting
            displacement = window_size // 2
            # cyclic shift is a faster way of adding padding
            # it is done by taking the area that is left out and pasting it to the bottom and right
            self.cyclic_shift = CyclicShift(-displacement)
            # shift back
            self.cyclic_back_shift = CyclicShift(displacement)
            # masking to mask the shifted parts after cyclic shift, so they are not related to each other
            # exponential function -> -infinity -> when we do not want relationship there
            # requires_grad is false because these are not learnable parameters
            self.upper_lower_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                             upper_lower=True, left_right=False), requires_grad=False)
            self.left_right_mask = nn.Parameter(create_mask(window_size=window_size, displacement=displacement,
                                                            upper_lower=False, left_right=True), requires_grad=False)
        # *3 because of query, key, value
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        if self.relative_pos_embedding:
            # creating the big matrix to put parameters in
            self.relative_indices = get_relative_distances(window_size) + window_size - 1
            # window_size - 1 -> because I have window size of 7 and that means that each pixel has 6 possible relationships
            # (13, 13) -> 6 on the right (up) nad 6 on the left (down) + itself
            self.pos_embedding = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        else:
            # absolute
            # when we divide image into patches, we need to keep information where that patch belongs
            self.pos_embedding = nn.Parameter(torch.randn(window_size ** 2, window_size ** 2))

        # inner_dim and dim are the same and equal to the number of channels ?
        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x):
        if self.shifted:
            # does not change the size of tensor
            x = self.cyclic_shift(x)

        # batch, height, width, heads
        b, n_h, n_w, _, h = *x.shape, self.heads

        # splits tensor into chunks or views along dimension dim (class dimension in this case)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # how many windows do we have in each stage
        nw_h = n_h // self.window_size
        nw_w = n_w // self.window_size

        q, k, v = map(
            # w_w and w_h are window sizes
            # input -> rearrange to output
            # d = head_dim - embedding dimension
            lambda t: rearrange(t, 'b (nw_h w_h) (nw_w w_w) (h d) -> b h (nw_h nw_w) (w_h w_w) d',
                                h=h, w_h=self.window_size, w_w=self.window_size), qkv)

        # dot product similarity
        # batch, heads, num of windows, i = 49 = j (window_size?), d = 32 (head_dim?)
        # reshaping ??
        # b h w j d   is the dimension of k
        # first two are inputs and after arrow is output ?
        # QK transpose inside softmax
        # dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.scale

        # instead of dot product similarity above we are going to use cosine

        # first we need to normalize q and k with respect to each row (p=2)
        q = f.normalize(q, p=2, dim=-1)
        k = f.normalize(k, p=2, dim=-1)
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) / self.tau



        # now we add B as positional embedding
        if self.relative_pos_embedding:
            # size of window size to the power of two but has depth of 2 -> 2 channels
            # describes how each row is connected to each other
            # indices of first channel and indices of second channel
            # they have to be valid indexes
            dots += self.pos_embedding[self.relative_indices[:, :, 0], self.relative_indices[:, :, 1]]
        else:
            # adding this to all windows
            # size of window size to the power of two
            dots += self.pos_embedding

        # now we need to add masking when SMSA
        if self.shifted:
            # adding the mas only to the last row
            # first dim is batch size, second one is head and the third one is related to the number of windows
            dots[:, :, -nw_w:] += self.upper_lower_mask
            # adding the mas only to the last column
            dots[:, :, nw_w - 1::nw_w] += self.left_right_mask

        # now we have attention, size is identical to the size of the input
        attn = dots.softmax(dim=-1)

        # we need to multiply the softmax with our value
        # shape of attention, shape of value -> shape of output
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        # getting the shape of transformer input
        # stuff in the brackets is multiplied together
        out = rearrange(out, 'b h (nw_h nw_w) (w_h w_w) d -> b (nw_h w_h) (nw_w w_w) (h d)',
                        h=h, w_h=self.window_size, w_w=self.window_size, nw_h=nw_h, nw_w=nw_w)
        # not changing shape, just linear layer
        out = self.to_out(out)

        if self.shifted:
            out = self.cyclic_back_shift(out)
        return out




class SwinBlock(nn.Module):
    def __init__(self, dim, heads, head_dim, mlp_dim, shifted, window_size, relative_pos_embedding):
        super().__init__()
        self.attention_block = Residual(PostNorm(dim, WindowAttention(dim=dim,
                                                                      heads=heads,
                                                                      head_dim=head_dim,
                                                                      shifted=shifted,
                                                                      window_size=window_size,
                                                                      relative_pos_embedding=relative_pos_embedding)))
        self.mlp_block = Residual(PostNorm(dim, FeedForward(dim=dim, hidden_dim=mlp_dim)))

    def forward(self, x):
        x = self.attention_block(x)
        x = self.mlp_block(x)
        return x





class StageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, downscaling_factor, num_heads, head_dim, window_size,
                 relative_pos_embedding, with_split, reduce_channels_to= None, embed=False):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'
        self.embed = embed
        if embed:
            patch_size = downscaling_factor
            self.embedding = SwinEmbedding(in_channels, hidden_dimension, patch_size)
        else:
            # change size and create hierarchy -> to use as an input to transformer block
            self.patch_partition = PatchMerging(in_channels=in_channels, out_channels=hidden_dimension,
                                                downscaling_factor=downscaling_factor)
        self.with_split = with_split
        if self.with_split:
            self.split = SplitUnit(in_channels=hidden_dimension, hidden_channels=reduce_channels_to)

        self.layers = nn.ModuleList([])
        # input and output size of transformer block is identical (the transformer itself does not change the dimensions)
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                # one block for Window MSA
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                # one block for Shifted Window MSA
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x, result_dict, exit_split):
        if self.embed:
            x = self.embedding(x)
        else:
            x = self.patch_partition(x)
        if self.with_split:
            x = self.split(x.permute(0, 3, 1, 2), result_dict)
            if exit_split:
                return x
            x = x.permute(0, 2, 3, 1)

        # print("Inside transformer after applying patch shape:", x.shape)
        # print("Inside transformer after applying patch size:", get_used_memory(x))
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        # return x.permute(0, 3, 1, 2)
        return x.permute(0, 3, 1, 2)
    # B C H W



class SkipDecoderStageModule(nn.Module):
    def __init__(self, in_channels, hidden_dimension, layers, num_heads, head_dim, window_size,
                 relative_pos_embedding):
        super().__init__()
        assert layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.patch_expansion = PatchExpansion(in_channels=in_channels)
        # output channels should be equal to hidden_dimensions, should be reduced by factor of 2
        self.skip_linear = nn.Linear(hidden_dimension * 2, hidden_dimension)

        self.layers = nn.ModuleList([])
        # input and output size of transformer block is identical (the transformer itself does not change the dimensions)
        for _ in range(layers // 2):
            self.layers.append(nn.ModuleList([
                # one block for Window MSA
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=False, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
                # one block for Shifted Window MSA
                SwinBlock(dim=hidden_dimension, heads=num_heads, head_dim=head_dim, mlp_dim=hidden_dimension * 4,
                          shifted=True, window_size=window_size, relative_pos_embedding=relative_pos_embedding),
            ]))

    def forward(self, x, skip_connection):

        x = self.patch_expansion(x)
        skip_connection = skip_connection.permute(0, 2, 3, 1)
        x = torch.cat([x, skip_connection], dim=-1)
        x = self.skip_linear(x)
        for regular_block, shifted_block in self.layers:
            x = regular_block(x)
            x = shifted_block(x)
        return x.permute(0, 3, 1, 2)
    # B C H W


class SwinEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, patch_size):
        super().__init__()

        # patches not overlapping
        self.patch_merge = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=patch_size,
                                     stride=patch_size, padding=0)

    def forward(self, x):
        x = self.patch_merge(x).permute(0, 2, 3, 1)
        return x


class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)
        self.norm = nn.LayerNorm(in_channels * downscaling_factor ** 2)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        # B H W C
        x = self.norm(x)
        x = self.linear(x)
        return x


class PatchExpansion(nn.Module):

    def __init__(
            self,
            in_channels
    ):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels // 2)
        self.expand = nn.Linear(in_channels, 2 * in_channels, bias=False)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.expand(x)
        B, H, W, C = x.shape

        x = x.view(B, H, W, 2, 2, C // 4)
        x = x.permute(0, 1, 3, 2, 4, 5)

        x = x.reshape(B, H * 2, W * 2, C // 4)
        x = self.norm(x)
        return x


class FinalPatchExpansion(nn.Module):

    def __init__(
            self,
            out_channels
    ):
        super().__init__()
        self.norm = nn.LayerNorm(out_channels)
        self.expand = nn.Linear(out_channels, 16 * out_channels, bias=False)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.expand(x)


        B, H, W, C = x.shape

        x = x.view(B, H, W, 4, 4, C // 16)

        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(B, H * 4, W * 4, C // 16)
        x = self.norm(x)
        return x

# class PatchExpand(nn.Module):
#     # dim is probably input dim
#     def __init__(self, input_resolution, in_channels, dim_scale=2, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = in_channels
#         self.expand = nn.Linear(in_channels, 2 * in_channels, bias=False) if dim_scale == 2 else nn.Identity()
#         self.norm = norm_layer(in_channels // dim_scale)
#
#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#         H, W = self.input_resolution
#         print(x.shape)
#         x = self.expand(x)
#         print(x.shape)
#         print(x.shape.permute(0, 2, 3, 1))
#         exit('Exited in patch expand')
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#
#         x = x.view(B, H, W, C)
#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
#         x = x.view(B,-1,C//4)
#         x= self.norm(x)
#
#         return x
#
# class FinalPatchExpand_X4(nn.Module):
#     # dim is how many channels should output have
#     def __init__(self, input_resolution, out_channels, dim_scale=4, norm_layer=nn.LayerNorm):
#         super().__init__()
#         self.input_resolution = input_resolution
#         self.dim = out_channels
#         self.dim_scale = dim_scale
#         self.expand = nn.Linear(out_channels, 16 * out_channels, bias=False)
#         self.output_dim = out_channels
#         self.norm = norm_layer(self.output_dim)
#
#     def forward(self, x):
#         """
#         x: B, H*W, C
#         """
#         H, W = self.input_resolution
#         print(x.shape)
#         x = self.expand(x)
#         print(x.shape)
#         print(x.shape.permute(0, 2, 3, 1))
#         exit('Exited in patch expand final')
#         B, L, C = x.shape
#         assert L == H * W, "input feature has wrong size"
#
#         x = x.view(B, H, W, C)
#         x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
#         x = x.view(B,-1,self.output_dim)
#         x = self.norm(x)
#
#         return x
