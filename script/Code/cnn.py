from torch import nn
import torch
from torchsummary import summary

class CNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
         ############### 1ST PARALLEL 2D CONVOLUTION BLOCK ############
        # 3 sequential conv2D layers: (1,40,282) --> (16, 20, 141) -> (32, 5, 35) -> (64, 1, 8)
        self.conv2Dblock1 = nn.Sequential(
            
            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1, # input volume depth == input channel dim == 1
                out_channels=16, # expand output feature map volume's depth to 16
                kernel_size=3, # typical 3*3 stride 1 kernel
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(16), # batch normalize the output feature map before activation
            nn.ReLU(), # feature map --> activation map
            nn.MaxPool2d(kernel_size=2, stride=2), #typical maxpool kernel size
            nn.Dropout(p=0.3), #randomly zero 30% of 1st layer's output feature map in training
            
            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, # expand output feature map volume's depth to 32
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4), # increase maxpool kernel for subsequent filters
            nn.Dropout(p=0.3), 
            
            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64, # expand output feature map volume's depth to 64
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            
            # 4th 2D convolution layer
        #     nn.Conv2d(
        #         in_channels=64, # input volume depth == input channel dim == 1
        #         out_channels=128, # expand output feature map volume's depth to 16
        #         kernel_size=3, # typical 3*3 stride 1 kernel
        #         stride=1,
        #         padding=1
        #               ),
        #     nn.BatchNorm2d(128), # batch normalize the output feature map before activation
        #     nn.ReLU(), # feature map --> activation map
        #     nn.MaxPool2d(kernel_size=2, stride=2), #typical maxpool kernel size
        #     nn.Dropout(p=0.3), #randomly zero 30% of 1st layer's output feature map in training
            
        #     # 5th 2D convolution layer identical to last except output dim, maxpool kernel
        #     nn.Conv2d(
        #         in_channels=128, 
        #         out_channels=256, # expand output feature map volume's depth to 32
        #         kernel_size=3,
        #         stride=1,
        #         padding=1
        #               ),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2), # increase maxpool kernel for subsequent filters
        #     nn.Dropout(p=0.3), 
            
        #     # 6th 2D convolution layer identical to last except output dim
        #     nn.Conv2d(
        #         in_channels=256,
        #         out_channels=512, # expand output feature map volume's depth to 64
        #         kernel_size=3,
        #         stride=1,
        #         padding=1
        #               ),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        #     nn.Dropout(p=0.3),
        )
         ################# FINAL LINEAR BLOCK ####################
        # Linear softmax layer to take final concatenated embedding tensor 
        # from parallel 2D convolutional and transformer blocks, output 8 logits 
        # Each full convolution block outputs (64*1*8) embedding flattened to dim 512 1D array 
        # Full transformer block outputs 40*70 feature map, which we time-avg to dim 40 1D array
        # 512*2+40 == 1064 input features --> 8 output emotions 
        self.fc1_linear = nn.Linear(
                            in_features = 64*1*4,
                            out_features = 6
                            ) 
        
        ### Softmax layer for the 6 output logits from final FC linear layer 
        self.softmax_out = nn.Softmax(dim=1) # dim==1 is the freq embedding

    def forward(self, input_data):
        # create final feature embedding from 1st convolutional layer 
        # input features pased through 4 sequential 2D convolutional layers
        conv2d_embedding1 = self.conv2Dblock1(input_data) # x == N/batch * channel * freq * time
        # flatten final 64*1*8 feature map from convolutional layers to length 512 1D array 
        # skip the 1st (N/batch) dimension when flattening
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1) 

        output_logits = self.fc1_linear(conv2d_embedding1)  
        output_softmax = self.softmax_out(output_logits)

        return output_logits, output_softmax

if __name__ == "__main__":
    cnn = CNNNetwork()
    model = cnn.to("cuda")
    summary(model, (1, 40, 130))