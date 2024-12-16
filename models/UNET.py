from utils.ModelUtils import *

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNET, self).__init__()
        # storing conv layers
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # we have to be careful about flooring with uppooling 161x161 -> 160x160

        # DOWN part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # UP part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    # times two because of skip connection
                    # doubles the height and the width of an image
                    in_channels=feature*2, out_channels=feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(in_channels=feature*2, out_channels=feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        # reversing list
        skip_connections = skip_connections[::-1]

        # up and double conv
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            # because of idx is like times 2 so // 2
            skip_connection = skip_connections[idx//2]

            # because of rounding
            if x.shape != skip_connection.shape:
                # you can solve this by different methods like padding
                # we are going to resize
                # should not impact accuracy too much, it's just one pixel
                # just taking size not bs and number of channels
                x = F.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)