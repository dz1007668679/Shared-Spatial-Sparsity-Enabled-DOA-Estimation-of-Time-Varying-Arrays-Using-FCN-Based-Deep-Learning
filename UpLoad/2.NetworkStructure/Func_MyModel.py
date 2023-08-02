import torch

# ==========构建训练网络========================
class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=3, out_channels=45, kernel_size=31, padding='same',
                                     padding_mode='reflect')
        self.conv2 = torch.nn.Conv1d(in_channels=45, out_channels=108, kernel_size=31, padding='same',
                                     padding_mode='reflect')
        self.conv3 = torch.nn.Conv1d(in_channels=108, out_channels=256, kernel_size=31, padding='same',
                                     padding_mode='reflect')
        self.conv4 = torch.nn.Conv1d(in_channels=256, out_channels=108, kernel_size=31, padding='same',
                                     padding_mode='reflect')
        self.conv5 = torch.nn.Conv1d(in_channels=108 * 2, out_channels=45, kernel_size=31, padding='same',
                                     padding_mode='reflect')
        self.conv6 = torch.nn.Conv1d(in_channels=45 * 2, out_channels=3, kernel_size=31, padding='same',
                                     padding_mode='reflect')
        self.conv7 = torch.nn.Conv1d(in_channels=3 * 2, out_channels=1, kernel_size=1, padding='same',
                                     padding_mode='reflect')
        self.batch_norm1 = torch.nn.BatchNorm1d(45)
        self.batch_norm2 = torch.nn.BatchNorm1d(108)
        self.batch_norm3 = torch.nn.BatchNorm1d(256)
        self.batch_norm4 = torch.nn.BatchNorm1d(108)
        self.batch_norm5 = torch.nn.BatchNorm1d(45)
        self.batch_norm6 = torch.nn.BatchNorm1d(3)
        self.batch_norm7 = torch.nn.BatchNorm1d(1)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.batch_norm1(x1)
        x1 = torch.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.batch_norm2(x2)
        x2 = torch.relu(x2)

        x3 = self.conv3(x2)
        x3 = self.batch_norm3(x3)
        x3 = torch.relu(x3)

        x4 = self.conv4(x3)
        x4 = self.batch_norm4(x4)
        x4 = torch.relu(x4)
        x4 = torch.cat((x2, x4), 1)

        x5 = self.conv5(x4)
        x5 = self.batch_norm5(x5)
        x5 = torch.relu(x5)
        x5 = torch.cat((x1, x5), 1)

        x6 = self.conv6(x5)
        x6 = self.batch_norm6(x6)
        x6 = torch.relu(x6)
        x6 = torch.cat((x, x6), 1)

        x7 = self.conv7(x6)
        x7 = self.batch_norm7(x7)
        x7 = torch.sigmoid(x7)

        x7 = torch.squeeze(x7, dim=1)

        return x7