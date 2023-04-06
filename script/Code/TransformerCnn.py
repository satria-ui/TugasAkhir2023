from torch import nn
import torch
from torchsummary import summary

class TransformerCNNNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        ################ TRANSFORMER BLOCK #############################
        # a rectangular kernel worked better here for the rectangular input spectrogram feature map/tensor
        self.transformer_maxpool = nn.MaxPool2d(kernel_size=[1,4], stride=[1,4])
        
        # define single transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=40,
            nhead=8, 
            dim_feedforward=512, 
            activation='relu',
            dropout=0.4
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)

        ############### 1ST PARALLEL 2D CONVOLUTION BLOCK ############
        self.conv2Dblock1 = nn.Sequential(
            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1,
                out_channels=16, 
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.ELU(), 
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
            
            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16, 
                out_channels=32, 
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3),
            
            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64, 
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3))
        
        ############### 2ND PARALLEL 2D CONVOLUTION BLOCK ############
        self.conv2Dblock2 = nn.Sequential(
            # 1st 2D convolution layer
            nn.Conv2d(
                in_channels=1, 
                out_channels=16, 
                kernel_size=3, 
                stride=1,
                padding=1
                      ),
            nn.ELU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3), 
            
            # 2nd 2D convolution layer identical to last except output dim, maxpool kernel
            nn.Conv2d(
                in_channels=16, 
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1
                      ),    
            nn.ELU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=4, stride=4), 
            nn.Dropout(p=0.3),
            # 3rd 2D convolution layer identical to last except output dim
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
                      ),
            nn.ELU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Dropout(p=0.3))
        
        ################# FINAL LINEAR BLOCK ####################
        self.fc1_linear = nn.Linear(
                            in_features = ((64*1*4)*2)+40,
                            out_features = 6
                            ) 
        
        ### Softmax layer for the 6 output logits from final FC linear layer 
        self.softmax_out = nn.Softmax(dim=1) # dim==1 is the freq embedding

    def forward(self, input_data):
        conv2d_embedding1 = self.conv2Dblock1(input_data)
        conv2d_embedding1 = torch.flatten(conv2d_embedding1, start_dim=1)
        conv2d_embedding2 = self.conv2Dblock2(input_data)
        conv2d_embedding2 = torch.flatten(conv2d_embedding2, start_dim=1) 

        x_maxpool = self.transformer_maxpool(input_data)

        # remove channel dim: 1*40*21 --> 40*21
        x_maxpool_reduced = torch.squeeze(x_maxpool,1)
        
        # convert maxpooled feature map format: batch * freq * time ---> time * batch * freq format
        # because transformer encoder layer requires tensor in format: time * batch * embedding (freq)
        input_data = x_maxpool_reduced.permute(2,0,1) 
        
        # finally, pass reduced input feature map x into transformer encoder layers
        transformer_output = self.transformer_encoder(input_data)
        
        # create final feature emedding from transformer layer by taking mean in the time dimension (now the 0th dim)
        # transformer outputs 2x40 (MFCC embedding*time) feature map, take mean of columns i.e. take time average
        transformer_embedding = torch.mean(transformer_output, dim=0) # dim 40x21 --> 40

        complete_embedding = torch.cat([conv2d_embedding1, conv2d_embedding2, transformer_embedding], dim=1)  

        output_logits = self.fc1_linear(complete_embedding)  
        output_softmax = self.softmax_out(output_logits)

        return output_logits, output_softmax

if __name__ == "__main__":
    cnn = TransformerCNNNetwork()
    model = cnn.to("cuda")
    summary(model, (1, 40, 615))