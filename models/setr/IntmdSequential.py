import torch.nn as nn


class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        print('In intermediate')
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            print('In loop', name)
            output = intermediate_outputs[name] = module(output)
            print('Output shape', output.shape)

        return output, intermediate_outputs
