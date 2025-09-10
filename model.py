import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNRegressor(nn.Module):

    """
      Convolutional Neural Network for crop yield prediction

      Architecture :
        - Three convolutional layers with batch normalization and max pooling
        - Dropout for regularization
        - Two fully connected layers with dropout

        The network expects input images of shape (batch_size, channels, 253, 253)
        and outputs a single yield prediction per image.
    """

    def __init__(self, input_channels=5, input_size = [140,140], dropout_rate=0.2):

        """
        Initialize the CNN regressor

        Args :
          input_channels (int, optional) : Number of input channels (default: 5)
          dropout_rate (float, optional) : Dropout rate for regularization (default: 0.2)

        """

        super(CNNRegressor, self).__init__()

        #First conlutional block
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        #Second convolutional block
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        #Third convolutional block
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        #Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(p=dropout_rate)


        self.flatten = nn.Flatten()

        conv_output_size = input_size[0] // 8
        fc_input_size = 128 * conv_output_size * conv_output_size

        self.fc1 = nn.Linear(fc_input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        self.fc_dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        """
          Forward pass through the network

          Args :
            x (torch.Tensor) : Input tensor of shape (batch_size, channels, 253, 253)

          Returns :
            torch.Tensor : Output tensor of shape (batch_size, 1)
        """

        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout(x)

        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)


        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc_dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc_dropout(x)
        x = self.fc3(x)

        return x.squeeze(1)