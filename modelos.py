import torch.nn as nn
import torch.nn.functional as F


def create_global_model(input_shape, num_classes):
    class GlobalModel(nn.Module):
        def __init__(self):
            super(GlobalModel, self).__init__()
            self.fc1 = nn.Linear(input_shape, 128)
            self.bn1 = nn.BatchNorm1d(128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, num_classes)  # Cambiado a 7 neuronas
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = F.relu(self.bn1(self.fc1(x)))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.fc3(x)  # softmax logarítmico para multi-clase
            return x

    return GlobalModel()

