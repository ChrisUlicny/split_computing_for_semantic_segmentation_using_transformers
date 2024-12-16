import torch.nn as nn
from utils.ModelUtils import AvgUnpoolingLayer


# class PoolingStrategyName(Enum):
#
#     MaxPooling = 0
#     AvgPooling = 1
#     ConvPooling = 2


class PoolingStrategy:

    def do_pooling(self, inputs, layer=None):
        raise NotImplementedError()

    def do_unpooling(self, inputs, indices, output_size, layer=None):
        raise NotImplementedError()

    def has_parameters(self):
        raise NotImplementedError()


class MaxPoolingStrategy(PoolingStrategy):

    def __init__(self):
        super().__init__()
        self.pooling_layer = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpooling_layer = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def do_pooling(self, inputs, layer=None):
        pooled_output, ind = self.pooling_layer(inputs)
        return pooled_output, ind

    def do_unpooling(self, inputs, indices, output_size, layer=None):
        unpooled_output = self.unpooling_layer(input=inputs, indices=indices, output_size=output_size)
        return unpooled_output

    def has_parameters(self):
        return False


class AvgPoolingStrategy(PoolingStrategy):

    def __init__(self):
        super().__init__()
        self.pooling_layer = nn.AvgPool2d(kernel_size=2, stride=2)
        self.unpooling_layer = AvgUnpoolingLayer()

    def do_pooling(self, inputs, layer=None):
        pooled_output = self.pooling_layer(inputs)
        return pooled_output, 0

    def do_unpooling(self, inputs, indices, output_size, layer=None):
        unpooled_output = self.unpooling_layer(inputs=inputs, output_size=output_size)
        return unpooled_output

    def has_parameters(self):
        return False


class ConvPoolingStrategy(PoolingStrategy):

    def do_pooling(self, inputs, layer=None):
        # channels = inputs.shape[1]
        # print("Determined channels:", channels, "\n")
        # pooling_layer = nn.Conv2d(channels, channels, kernel_size=2, stride=2)
        # pooled_output = pooling_layer(inputs)
        pooled_output = layer(inputs)
        # print("Pooled size: {}".format(pooled_output.shape))
        return pooled_output, 0

    def do_unpooling(self, inputs, indices, output_size, layer=None):
        # Calculate output_padding
        # output_padding = calculate_output_padding(input_size=inputs.size(),
        #                                           output_size=output_size,
        #                                           stride=2,
        #                                           kernel_size=2)
        # layer.output_padding = output_padding
        unpooled_output = layer(inputs)
        return unpooled_output

    def has_parameters(self):
        return True


def calculate_output_padding(input_size, output_size, stride, kernel_size):
    # Calculate the expected output size without output_padding
    expected_output_size_h = (input_size[2] - 1) * stride + kernel_size
    expected_output_size_w = (input_size[3] - 1) * stride + kernel_size

    # Calculate the needed output_padding to match the desired output size
    output_padding_h = max(0, output_size[2] - expected_output_size_h)
    output_padding_w = max(0, output_size[3] - expected_output_size_w)

    return (output_padding_h, output_padding_w)


