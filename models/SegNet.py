from utils.ModelUtils import *
from utils.utils import *
import metrics
import torch.optim as optim


class SegNet(nn.Module, GeneralModel):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512, 512]):
        super(SegNet, self).__init__()
        # storing conv layers
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.num_classes = out_channels
        # self.exit_ups = nn.ModuleList()
        # self.exit_early = False
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # DOWN part of SegNet
        counter = 1
        for feature in features:
            if counter < 3:
                self.downs.append(DoubleConv(in_channels, feature))
            else:
                self.downs.append(TripleConv(in_channels, feature))
            in_channels = feature
            counter += 1

        # UP part of SegNet
        self.ups.append(TripleConv(in_channels=features[4], out_channels=features[3]))
        self.ups.append(TripleConv(in_channels=features[3], out_channels=features[2]))
        self.ups.append(TripleConv(in_channels=features[2], out_channels=features[1]))
        self.ups.append(DoubleConv(in_channels=features[1], out_channels=features[0]))
        self.final_conv = nn.Sequential(
            nn.Conv2d(features[0], features[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(features[0], out_channels, kernel_size=3, padding=1)
        )
        self.ups.append(self.final_conv)

    def forward(self, x):
        # encoding
        indices, sizes = [], []
        input_size = x.size()
        for down in self.downs:
            x = down(x)
            x, ind = self.pool(x)
            indices.append(ind)
            sizes.append(x.size())

        # flipping for decoding
        indices = indices[::-1]
        sizes = sizes[::-1]

        # decoding
        for idx in range(0, len(self.ups)-1):
            output_size = sizes[idx+1]
            x = self.unpool(input=x, indices=indices[idx], output_size=output_size)
            x = self.ups[idx](x)

        idx = len(self.ups)-1
        output_size = input_size
        x = self.unpool(x, indices[idx], output_size=output_size)
        x = self.final_conv(x)
        return x

    def initialize_optimizer(self, learning_rate, optimizer_type):
        if optimizer_type == 'Adam':
            return optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_type == 'SGD':
            return optim.SGD(self.parameters(), lr=learning_rate)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.load_state_dict(state_dict=checkpoint["state_dict"], strict=False)
        epoch = checkpoint["epoch"]
        return epoch

    def check_batch_acc_with_loss(self, loader, loss_func=None):
        return metrics.check_batch_acc_miou_loss(self, loader, loss_func)

    def check_batch_miou(self, loader, num_classes=None):
        return metrics.check_batch_miou(loader, self, self.num_classes)

    def get_loss_func(self):
        return nn.CrossEntropyLoss()

    def set_strategy(self, strategy):
        pass

    def make_prediction(self, image, mask, model, idx, color_map, prediction_folder):
        if not os.path.exists(prediction_folder):
            # if the directory is not present, create it
            os.makedirs(prediction_folder)
        save_image(image, f"{prediction_folder}/original{idx}.png")
        mask = torch.argmax(mask, dim=3)
        mask = seg_map_to_image(mask, color_map)
        save_numpy_as_image(mask, f"{prediction_folder}/ground{idx}.png")
        prediction_for_accuracy = []
        start = time.time()
        prediction = model(image)
        end = time.time()
        prediction = torch.squeeze(prediction)
        prediction_for_accuracy.append(prediction)
        # print('prediction size pre argmax after model:', prediction.shape)
        prediction = torch.argmax(prediction, dim=0)
        # print('prediction size:', prediction.shape)
        new_image = seg_map_to_image(prediction, color_map)
        # print('new image size:', new_image.shape)
        save_numpy_as_image(new_image, f"{prediction_folder}/pred{idx}.png")
        time_elapsed = end - start
        return prediction_for_accuracy, time_elapsed

