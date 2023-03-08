from torch import nn
import torch
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv2Dblock1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=6,
                kernel_size=5, 
                stride=1,
                padding=0
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
                in_channels=6, 
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0
                      ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc1_linear = nn.Linear(in_features = 16*7*18,out_features = 120) 
        self.fc2_linear = nn.Linear(in_features = 120,out_features = 84) 
        self.fc3_linear = nn.Linear(in_features = 84,out_features = 6) 
        
        ### Softmax layer for the 6 output logits from final FC linear layer 
        self.softmax_out = nn.Softmax(dim=1)


        # # convolutional layers
        # self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        # self.conv2 = nn.Conv2d(6, 16, kernel_size=2, stride=1, padding=0)
        # # self.conv3 = nn.Conv2d(16, 120, kernel_size=2, stride=1, padding=0)
        
        # # subsampling layers
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # # fully connected layers
        # self.fc1 = nn.Linear(in_features = 16 * 10 * 21, out_features = 120)
        # self.fc2 = nn.Linear(in_features = 120, out_features = 84)
        # self.fc3 = nn.Linear(84, 6)
        
        # # activation function
        # self.relu = nn.ReLU()

    def forward(self, input_data):
        conv2d_embedding1 = self.conv2Dblock1(input_data)
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1) 

        linear = self.fc1_linear(conv2d_embedding1) 
        linear = self.fc2_linear(linear) 
        output_logits = self.fc3_linear(linear) 
        output_softmax = self.softmax_out(output_logits)

        return output_logits, output_softmax
        # input_data = self.relu(self.conv1(input_data))
        # input_data = self.pool1(input_data)
        # input_data = self.relu(self.conv2(input_data))
        # input_data = self.pool2(input_data)
        # # input_data = self.relu(self.conv3(input_data))

        # # input_data = input_data.reshape(input_data.shape[0], -1)
        # input_data = input_data.view(-1, 16 * 10 * 21)
        # input_data = self.relu(self.fc1(input_data))
        
        # output_logits = self.relu(self.fc2(input_data))
        # output_prediction = self.fc3(input_data)

        # return output_logits, output_prediction

if __name__ == "__main__":
    cnn = LeNet()
    model = cnn.to("cuda")
    summary(model, (1, 40, 87))