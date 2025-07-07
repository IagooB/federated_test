from confg.configuracion import *
import torch.nn as nn
import torch.nn.functional as F


def create_global_model(input_shape, num_classes):
    class GlobalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_shape, 64)
            self.norm1 = nn.LayerNorm(64)
            self.fc2 = nn.Linear(64, 32)
            self.norm2 = nn.LayerNorm(32)
            # skip connection
            self.fc_skip = nn.Linear(input_shape, 64, bias=False)
            self.fc_out = nn.Linear(32, num_classes)
            self.dropout = nn.Dropout(0.2)

        def forward(self, x):
            h1 = F.relu(self.norm1(self.fc1(x)))
            x = h1 + self.fc_skip(x)  # residual
            x = self.dropout(x)
            x2 = F.relu(self.norm2(self.fc2(x)))
            x2 = self.dropout(x2)
            return self.fc_out(x2)

    return GlobalModel()


"""def create_global_model(input_shape, num_classes):
    class GlobalModel(nn.Module):
        def __init__(self):
            super(GlobalModel, self).__init__()
            self.fc1 = nn.Linear(input_shape, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_classes)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)
            return x

    return GlobalModel()
"""
