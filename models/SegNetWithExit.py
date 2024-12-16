from utils.ModelUtils import *


class SegNetWithExit(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 512]):
        super(SegNetWithExit, self).__init__()
        # storing conv layers
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.exit_ups = nn.ModuleList()
        self.threshold = 0.55
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.inf = False
        self.training_backbone = False
        self.early_exit_count = 0
        self.final_exit_count = 0

        # DOWN part of SegNet
        counter = 1
        for feature in features:
            if counter < 3:
                self.downs.append(DoubleConv(in_channels, feature))
            else:
                self.downs.append(TripleConv(in_channels, feature))
            in_channels = feature
            counter += 1

        # UP part of exit - AlexNet after third conv
        # reducing channels instead first layer (Conv + RELU)
        self.exit_ups.append(SingleConv(in_channels=512, out_channels=512))
        self.exit_ups.append(SingleConv(in_channels=512, out_channels=256))
        self.exit_ups.append(SingleConv(in_channels=256, out_channels=128))
        self.exit_ups.append(SingleConv(in_channels=128, out_channels=64, kernel_size=5, padding=2))
        # self.exit_ups.append(SingleConv(in_channels=96, out_channels=out_channels, kernel_size=11, stride=4, padding=2))
        self.final_exit_conv = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=3, padding=1)

        # UP part of SegNet
        self.ups.append(TripleConv(in_channels=features[4], out_channels=features[3]))
        self.ups.append(TripleConv(in_channels=features[3], out_channels=features[2]))
        self.ups.append(TripleConv(in_channels=features[2], out_channels=features[1]))
        self.ups.append(DoubleConv(in_channels=features[1], out_channels=features[0]))
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True)
        )
        self.ups.append(self.final_conv)

    # removed last down sampling and first up sampling -> was useless
    def forward(self, x):
        if not self.inf:
            # encoding
            indices, sizes = [], []
            for idx, down in enumerate(self.downs):
                x = down(x)
                # disabling last pooling layer
                if idx < 4:
                    sizes.append(x.size())
                    x, ind = self.pool(x)
                    indices.append(ind)


            # flipping for decoding
            indices = indices[::-1]
            sizes = sizes[::-1]
            # cutting last added size which is not used anyway

            # early exit -> AlexNet-like
            if not self.training_backbone:
                # reducing channels 512 -> 256
                x = self.exit_ups[0](x)
                # first unpool
                output_size = sizes[0]
                x = self.unpool(input=x, indices=indices[0], output_size=output_size)
                x = self.exit_ups[1](x)
                output_size = sizes[1]
                x = self.unpool(input=x, indices=indices[1], output_size=output_size)
                x = self.exit_ups[2](x)
                output_size = sizes[2]
                x = self.unpool(input=x, indices=indices[2], output_size=output_size)
                x = self.exit_ups[3](x)
                output_size = sizes[3]
                x = self.unpool(input=x, indices=indices[3], output_size=output_size)
                x = self.final_exit_conv(x)
                return x

            else:
                # decoding part of segnet
                x = self.ups[0](x)
                # 2nd block
                output_size = sizes[0]
                x = self.unpool(input=x, indices=indices[0], output_size=output_size)
                x = self.ups[1](x)
                # 3rd block
                output_size = sizes[1]
                x = self.unpool(input=x, indices=indices[1], output_size=output_size)
                x = self.ups[2](x)
                # 4th block
                output_size = sizes[2]
                x = self.unpool(input=x, indices=indices[2], output_size=output_size)
                x = self.ups[3](x)
                # 5th block
                output_size = sizes[3]
                x = self.unpool(input=x, indices=indices[3], output_size=output_size)
                x = self.final_conv(x)
                return x

        # inference
        elif self.inf:
            # encoding
            indices, sizes = [], []
            for idx, down in enumerate(self.downs):
                x = down(x)
                # disabling last pooling layer
                if idx < 4:
                    sizes.append(x.size())
                    x, ind = self.pool(x)
                    indices.append(ind)


            # flipping for decoding
            indices = indices[::-1]
            sizes = sizes[::-1]

            # early exit -> AlexNet-like
            x_exit = self.exit_ups[0](x)
            # first unpool
            output_size = sizes[0]
            x_exit = self.unpool(input=x_exit, indices=indices[0], output_size=output_size)
            x_exit = self.exit_ups[1](x_exit)
            output_size = sizes[1]
            x_exit = self.unpool(input=x_exit, indices=indices[1], output_size=output_size)
            x_exit = self.exit_ups[2](x_exit)
            output_size = sizes[2]
            x_exit = self.unpool(input=x_exit, indices=indices[2], output_size=output_size)
            x_exit = self.exit_ups[3](x_exit)
            output_size = sizes[3]
            x_exit = self.unpool(input=x_exit, indices=indices[3], output_size=output_size)
            x_exit = self.final_exit_conv(x_exit)
            # calculating confidence for the prediction
            confidence = calculate_confidence(x_exit, self.threshold)
            if self.threshold < confidence:
                self.early_exit_count += 1
                # return x_exit

            # decoding part of segnet
            x = self.ups[0](x)
            # 2nd block
            output_size = sizes[0]
            x = self.unpool(input=x, indices=indices[0], output_size=output_size)
            x = self.ups[1](x)
            # 3rd block
            output_size = sizes[1]
            x = self.unpool(input=x, indices=indices[1], output_size=output_size)
            x = self.ups[2](x)
            # 4th block
            output_size = sizes[2]
            x = self.unpool(input=x, indices=indices[2], output_size=output_size)
            x = self.ups[3](x)
            # 5th block
            output_size = sizes[3]
            x = self.unpool(input=x, indices=indices[3], output_size=output_size)
            x = self.final_conv(x)
            self.final_exit_count += 1
            # return x
            return x, x_exit

    def print_exits_taken(self):
        print(f'Exited early: {self.early_exit_count}\nTook final exit: {self.final_exit_count}')

    def initialize_optimizer(self, learning_rate):
        if self.training_backbone:
            print("Training backbone")
            optimizer = optim.Adam(chain(self.downs.parameters(), self.ups.parameters(),
                                         self.final_conv.parameters()), lr=learning_rate)
        else:
            print("Training exit")
            optimizer = optim.Adam(chain(self.exit_ups.parameters(),
                                         self.final_exit_conv.parameters()), lr=learning_rate)
        return optimizer
