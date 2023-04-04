from torch import nn
import torch
from torchsummary import summary

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv2Dblock1 = nn.Sequential(
            nn.Conv2d(
            # 1st 2D convolution layer
                in_channels=1,
                out_channels=6,
                kernel_size=5, 
                stride=1,
                padding=0
                      ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(
            # 2nd 2D convolution layer
                in_channels=6, 
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=0
                      ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc1_linear = nn.Linear(in_features = 16*7*74,out_features = 120) 
        self.fc2_linear = nn.Linear(in_features = 120,out_features = 84) 
        self.fc3_linear = nn.Linear(in_features = 84,out_features = 6) 
        
        ### Softmax layer for the 6 output logits from final FC linear layer 
        self.softmax_out = nn.Softmax(dim=1)

    def forward(self, input_data):
        conv2d_embedding1 = self.conv2Dblock1(input_data)
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1) 

        linear = self.fc1_linear(conv2d_embedding1) 
        linear = self.fc2_linear(linear) 
        output_logits = self.fc3_linear(linear) 
        output_softmax = self.softmax_out(output_logits)

        return output_logits, output_softmax

if __name__ == "__main__":
    cnn = LeNet()
    model = cnn.to("cuda")
    summary(model, (1, 40, 615))