import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Convolutional block consisting of Sequential operations:
      - 3x3 Conv
      - BatchNorm
      - ReLU
      - 3x3 Conv
      - BatchNorm
      - ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class RecurrentBlock(nn.Module):
    """
    Recurrent convolutional block consisting of:
      - 3x3 Conv
      - ReLU
      - 3x3 Conv
      - ReLU
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x


class AttentionBlock(nn.Module):
    """
    Attention block to compute attention between 
    encoder and decoder features.
    """
    def __init__(self, encoder_channels, decoder_channels):
        super().__init__()
        
        self.theta = nn.Conv3d(encoder_channels, encoder_channels // 2, kernel_size=1)
        self.phi = nn.Conv3d(decoder_channels, decoder_channels // 2, kernel_size=1)
        self.psi = nn.Conv3d(encoder_channels, encoder_channels // 2, kernel_size=1)
        self.out_conv = nn.Conv3d(encoder_channels // 2, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, encoder_x, decoder_x):
        theta = self.theta(encoder_x) 
        phi = self.phi(decoder_x)
        psi = self.psi(encoder_x)
        attn = self.relu(theta + phi + psi)
        attn = self.out_conv(attn)
        attn = self.sigmoid(attn)
        return encoder_x * attn.expand_as(encoder_x) + decoder_x
    
class AR2B_UNet(nn.Module):
    """
    Implementation of a 3D UNet model with attention blocks and recurrent block for 3D Image/Video segmentation.

    The model performs the following steps:

    1. Encoder:
        - Input is passed through a series of ConvBlocks 
            which are convolutional blocks consisting of Conv3d, BatchNorm3d, ReLU layers.
        - Max pooling is applied after each ConvBlock to downsample the features.
        
    2. Recurrent Block:
        - The output of the encoder is passed through a RecurrentBlock.
            This contains two convolutional layers to capture temporal dependencies.
            
    3. Decoder:
        - The output of the recurrent block is upsampled using transposed convolutions.
        - It is concatenated with features from the encoder at the same level.
        - AttentionBlock is applied on the concatenated features 
            to focus on relevant encoded features.
        - The attended features are passed through ConvBlocks to obtain the output.
        
    4. Final Convolution:
        - A final 1x1 Conv3d layer generates the pixel-wise predictions.
        
    Args:
        in_channels (int): Number of input channels
        num_classes (int): Number of classes for segmentation
        
    Returns:
        A PyTorch model for 3D image / video segmentation.
    """
    def __init__(self, in_channels, num_classes):
        super(AR2B_UNet, self).__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 1024)

        # Recurrent Block
        self.recurrent_block = RecurrentBlock(1024, 256)

        # Decoder
        self.att1 = AttentionBlock(512, 512)
        self.att2 = AttentionBlock(256, 256)
        self.att3 = AttentionBlock(128, 128)
        self.att4 = AttentionBlock(64, 64)

        self.dec1 = ConvBlock(512 + 512, 512)
        self.dec2 = ConvBlock(256 + 256, 256)
        self.dec3 = ConvBlock(128 + 128, 128)
        self.dec4 = ConvBlock(64 + 64, 64)

        self.upconv1 = nn.ConvTranspose3d(256, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)

        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        x = F.max_pool3d(e1, 2)
        e2 = self.enc2(x)
        x = F.max_pool3d(e2, 2)
        e3 = self.enc3(x)
        x = F.max_pool3d(e3, 2)
        e4 = self.enc4(x)
        x = F.max_pool3d(e4, 2)
        x = self.enc5(x)
        x = self.recurrent_block(x)

        x = self.upconv1(x)
        x = self.att1(e4, x)
        x = self.dec1(torch.cat((e4, x), dim=1))

        x = self.upconv2(x)
        x = self.att2(e3, x)
        x = self.dec2(torch.cat((e3, x), dim=1))

        x = self.upconv3(x)
        x = self.att3(e2, x)
        x = self.dec3(torch.cat((e2, x), dim=1))

        x = self.upconv4(x)
        x = self.att4(e1, x)
        x = self.dec4(torch.cat((e1, x), dim=1))

        x = self.final_conv(x)
        return x

class AR2B_UNet_DeepSup(nn.Module):
    """
    Implementation of a 3D UNet model with attention blocks, recurrent block, and deep supervision for 3D Image/Video segmentation.

    The model performs the following steps:

    1. Encoder:
        - Input is passed through a series of ConvBlocks 
            which are convolutional blocks consisting of Conv3d, BatchNorm3d, ReLU layers.
        - Max pooling is applied after each ConvBlock to downsample the features.

    2. Deep Supervision Branches:
        - After certain encoding steps, auxiliary classifiers are applied to 
            produce intermediate segmentations. These are used for deep supervision
            during training to aid in gradient propagation and potentially improve
            convergence.

    3. Recurrent Block:
        - The output of the deepest encoder is passed through a RecurrentBlock.
            This contains two convolutional layers to capture temporal dependencies.

    4. Decoder:
        - The output of the recurrent block is upsampled using transposed convolutions.
        - It is concatenated with features from the encoder at the same level.
        - AttentionBlock is applied on the concatenated features 
            to focus on relevant encoded features.
        - The attended features are passed through ConvBlocks to obtain the output.

    5. Final Convolution:
        - A final 1x1 Conv3d layer generates the pixel-wise predictions.

    During training, the model produces one main output and several auxiliary outputs 
    from the deep supervision branches. The combined loss from these outputs is used
    to update the model weights.

    Args:
        in_channels (int): Number of input channels
        num_classes (int): Number of classes for segmentation

    Returns:
        A PyTorch model for 3D image / video segmentation. During training, the model 
        produces multiple outputs (main and auxiliary) for deep supervision.
    """
    def __init__(self, in_channels, num_classes):
        super(AR2B_UNet_DeepSup, self).__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 1024)

        # Recurrent Block
        self.recurrent_block = RecurrentBlock(1024, 256)

        # Decoder
        self.att1 = AttentionBlock(512, 512)
        self.att2 = AttentionBlock(256, 256)
        self.att3 = AttentionBlock(128, 128)
        self.att4 = AttentionBlock(64, 64)

        self.dec1 = ConvBlock(512 + 512, 512)
        self.dec2 = ConvBlock(256 + 256, 256)
        self.dec3 = ConvBlock(128 + 128, 128)
        self.dec4 = ConvBlock(64 + 64, 64)

        self.upconv1 = nn.ConvTranspose3d(256, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)

        # Adding auxiliary classifiers for deep supervision
        self.aux_classifier1 = nn.Conv3d(512, num_classes, kernel_size=1)
        self.aux_classifier2 = nn.Conv3d(256, num_classes, kernel_size=1)
        self.aux_classifier3 = nn.Conv3d(128, num_classes, kernel_size=1)
        self.aux_classifier4 = nn.Conv3d(64, num_classes, kernel_size=1)

        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        x = F.max_pool3d(e1, 2)
        e2 = self.enc2(x)
        x = F.max_pool3d(e2, 2)
        e3 = self.enc3(x)
        x = F.max_pool3d(e3, 2)
        e4 = self.enc4(x)
        x = F.max_pool3d(e4, 2)
        x = self.enc5(x)
        x = self.recurrent_block(x)

        aux_output1 = self.aux_classifier1(e4)
        x = self.upconv1(x)
        x = self.att1(e4, x)
        x = self.dec1(torch.cat((e4, x), dim=1))

        aux_output2 = self.aux_classifier2(e3)
        x = self.upconv2(x)
        x = self.att2(e3, x)
        x = self.dec2(torch.cat((e3, x), dim=1))

        aux_output3 = self.aux_classifier3(e2)
        x = self.upconv3(x)
        x = self.att3(e2, x)
        x = self.dec3(torch.cat((e2, x), dim=1))

        aux_output4 = self.aux_classifier4(e1)
        x = self.upconv4(x)
        x = self.att4(e1, x)
        x = self.dec4(torch.cat((e1, x), dim=1))

        # Make aux classifiers produce outputs matching target size
        aux_output1 = F.interpolate(aux_output1, size=(128,128,128))
        aux_output2 = F.interpolate(aux_output2, size=(128,128,128))
        aux_output3 = F.interpolate(aux_output3, size=(128,128,128))
        aux_output4 = F.interpolate(aux_output4, size=(128,128,128))

        x = self.final_conv(x)
        return x, aux_output1, aux_output2, aux_output3, aux_output4
    

# class SwinTransformerBlock(nn.Module):
#     def __init__(self, in_channels, num_heads=4, window_size=4):
#         super(SwinTransformerBlock, self).__init__()
#         self.embed_dim = in_channels * window_size * window_size * window_size
#         self.num_heads = num_heads
#         self.head_dim = self.embed_dim // self.num_heads
#         self.window_size = window_size

#         # Ensure embed_dim is divisible by num_heads
#         assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

#         self.attention = nn.MultiheadAttention(self.embed_dim, self.num_heads)
#         self.mlp = nn.Sequential(
#             nn.Linear(self.embed_dim, self.embed_dim),
#             nn.ReLU(),
#             nn.Linear(self.embed_dim, self.embed_dim)
#         )

#     def forward(self, x):
#         # Split input into non-overlapping windows and flatten
#         B, C, D, H, W = x.shape
#         x = x.view(B, C, D // self.window_size, self.window_size, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
#         x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(B, D // self.window_size * H // self.window_size * W // self.window_size, C * self.window_size * self.window_size * self.window_size)

#         # Apply self-attention
#         attn_output, _ = self.attention(x, x, x)
#         x = x + attn_output

#         # Apply MLP
#         x = x + self.mlp(x)

#         # Reshape back to original shape
#         x = x.view(B, D // self.window_size, H // self.window_size, W // self.window_size, self.window_size, self.window_size, self.window_size, C).permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
#         x = x.view(B, C, D, H, W)
#         return x

class SwinTransformerBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads=4, window_size=4):
        super(SwinTransformerBlock, self).__init__()
        self.embed_dim = in_channels * window_size * window_size * window_size
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.window_size = window_size

        # Ensure embed_dim is divisible by num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.attention = nn.MultiheadAttention(self.embed_dim, self.num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.ReLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )
        self.conv1x1x1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Split input into non-overlapping windows and flatten
        B, C, D, H, W = x.shape
        x = x.view(B, C, D // self.window_size, self.window_size, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous().view(B, D // self.window_size * H // self.window_size * W // self.window_size, C * self.window_size * self.window_size * self.window_size)

        # Apply self-attention
        attn_output, _ = self.attention(x, x, x)
        x = x + attn_output

        # Apply MLP
        x = x + self.mlp(x)

        # Reshape back to original shape
        x = x.view(B, D // self.window_size, H // self.window_size, W // self.window_size, self.window_size, self.window_size, self.window_size, C).permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        x = x.view(B, C, D, H, W)

        # Transform channels to desired output channels
        x = self.conv1x1x1(x)
        return x

class Swin_AR2B_DeepSup_UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Swin_AR2B_DeepSup_UNet, self).__init__()

        # Encoder with Swin Transformer blocks
        self.enc1 = SwinTransformerBlock(in_channels = in_channels, out_channels= 64, window_size=2)
        self.enc2 = SwinTransformerBlock(64, out_channels= 128, window_size=2)
        self.enc3 = SwinTransformerBlock(128, out_channels= 256, window_size=2)
        self.enc4 = SwinTransformerBlock(256, out_channels= 512, window_size=2)
        self.enc5 = SwinTransformerBlock(512, out_channels= 1024, window_size=2)

        
        # Encoder with Swin Transformer blocks
        # self.enc1 = SwinTransformerBlock(in_channels, 2)
        # self.enc2 = SwinTransformerBlock(64, 2)
        # self.enc3 = SwinTransformerBlock(128, 2)
        # self.enc4 = SwinTransformerBlock(256, 2)
        # self.enc5 = SwinTransformerBlock(512, 2)

        # Recurrent Block
        #self.recurrent_block = RecurrentBlock(4, 256)
        self.recurrent_block = RecurrentBlock(1024, 256)

        # Decoder with Attention blocks
        self.att1 = AttentionBlock(512, 512)
        self.att2 = AttentionBlock(256, 256)
        self.att3 = AttentionBlock(128, 128)
        self.att4 = AttentionBlock(64, 64)

        self.dec1 = ConvBlock(512 + 512, 512)
        self.dec2 = ConvBlock(256 + 256, 256)
        self.dec3 = ConvBlock(128 + 128, 128)
        self.dec4 = ConvBlock(64 + 64, 64)

        self.upconv1 = nn.ConvTranspose3d(256, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)

        # Deep supervision classifiers
        self.aux_classifier1 = nn.Conv3d(512, num_classes, kernel_size=1)
        self.aux_classifier2 = nn.Conv3d(256, num_classes, kernel_size=1)
        self.aux_classifier3 = nn.Conv3d(128, num_classes, kernel_size=1)
        self.aux_classifier4 = nn.Conv3d(64, num_classes, kernel_size=1)

        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        #print(e1.shape)
        x = F.max_pool3d(e1, 2)
        #print(x.shape)
        e2 = self.enc2(x)
        #print(e2.shape)
        x = F.max_pool3d(e2, 2)
        #print(x.shape)
        e3 = self.enc3(x)
        #print(e3.shape)
        x = F.max_pool3d(e3, 2)
        #print(x.shape)
        e4 = self.enc4(x)
        #print(e4.shape)
        x = F.max_pool3d(e4, 2)
        #print(x.shape)
        x = self.enc5(x)
        #print(x.shape)
        x = self.recurrent_block(x)
        #print(x.shape)

        aux_output1 = self.aux_classifier1(e4)
        x = self.upconv1(x)
        x = self.att1(e4, x)
        x = self.dec1(torch.cat((e4, x), dim=1))

        aux_output2 = self.aux_classifier2(e3)
        x = self.upconv2(x)
        x = self.att2(e3, x)
        x = self.dec2(torch.cat((e3, x), dim=1))

        aux_output3 = self.aux_classifier3(e2)
        x = self.upconv3(x)
        x = self.att3(e2, x)
        x = self.dec3(torch.cat((e2, x), dim=1))

        aux_output4 = self.aux_classifier4(e1)
        x = self.upconv4(x)
        x = self.att4(e1, x)
        x = self.dec4(torch.cat((e1, x), dim=1))

        # Resize auxiliary outputs to match target size
        aux_output1 = F.interpolate(aux_output1, size=(128,128,128))
        aux_output2 = F.interpolate(aux_output2, size=(128,128,128))
        aux_output3 = F.interpolate(aux_output3, size=(128,128,128))
        aux_output4 = F.interpolate(aux_output4, size=(128,128,128))

        x = self.final_conv(x)
        return x, aux_output1, aux_output2, aux_output3, aux_output4


class AR2B_MultiStep_UNet(nn.Module):
    """
    Implementation of a multi-step segmentation model for brain tumours.

    The model first performs the following steps:

    1. Encoder: The input images are passed through a series of convolutional blocks to extract features.
    2. Recurrent Block: The extracted features are passed through a recurrent block to capture long-range dependencies.
    3. Decoder: The features from the recurrent block are upsampled and concatenated with the features from the encoder. This process is repeated several times until the original image size is reached.
    4. Classifiers for whole tumour detection and separate tumour classes: The final layer of the decoder is split into two branches. The first branch is used to classify the images into two classes: whole tumour (1) and background (0). The second branch is used to classify the images into three classes: necrosis (1), edema (2), and enhancing tumor (3).
    5. Final convolution layer: The output of the two classifiers is concatenated and passed through a final convolutional layer to generate the final segmentation mask.

    The model then performs the following steps to do multi-step segmentation:

    1. The whole tumour mask is used to remove the background from the images.
    2. The images without background are passed through the tumour classifier to segment the tumour into separate classes.
    3. The results from the two steps are combined to generate the final segmentation mask.

    Args:
        in_channels (int): The number of input channels.
        num_classes (int): The number of output classes.

    Returns:
        A PyTorch model for multi-step brain tumour segmentation.
    """
    def __init__(self, in_channels, num_classes):
        super(AR2B_MultiStep_UNet, self).__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)
        self.enc5 = ConvBlock(512, 1024)

        # Recurrent Block
        self.recurrent_block = RecurrentBlock(1024, 256)

        # Decoder
        self.att1 = AttentionBlock(512, 512)
        self.att2 = AttentionBlock(256, 256)
        self.att3 = AttentionBlock(128, 128)
        self.att4 = AttentionBlock(64, 64)

        self.dec1 = ConvBlock(512 + 512, 512)
        self.dec2 = ConvBlock(256 + 256, 256)
        self.dec3 = ConvBlock(128 + 128, 128)
        self.dec4 = ConvBlock(64 + 64, 64)

        self.upconv1 = nn.ConvTranspose3d(256, 512, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)

        # Classifier for whole tumour detection
        self.whole_tumour_classifier = nn.Conv3d(256, 1, kernel_size=1)

        # Classifier for separate tumour classes
        self.tumour_classifier = nn.Conv3d(512, 3, kernel_size=1)

        self.final_conv = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        x = F.max_pool3d(e1, 2)
        e2 = self.enc2(x)
        x = F.max_pool3d(e2, 2)
        e3 = self.enc3(x)
        x = F.max_pool3d(e3, 2)
        e4 = self.enc4(x)
        x = F.max_pool3d(e4, 2)
        x = self.enc5(x)
        x = self.recurrent_block(x)

        # Whole tumour detection
        whole_tumour_mask = self.whole_tumour_classifier(x)

        # Remove background from images using whole tumour mask
        x = x * whole_tumour_mask

        # Segment tumour into separate classes
        tumour_mask = self.tumour_classifier(x)

        x = self.upconv1(x)
        x = self.att1(e4, x)
        x = self.dec1(torch.cat((e4, x), dim=1))

        x = self.upconv2(x)
        x = self.att2(e3, x)
        x = self.dec2(torch.cat((e3, x), dim=1))

        x = self.upconv3(x)
        x = self.att3(e2, x)
        x = self.dec3(torch.cat((e2, x), dim=1))

        x = self.upconv4(x)
        x = self.att4(e1, x)
        x = self.dec4(torch.cat((e1, x), dim=1))

        x = self.final_conv(x)

        # Combine the results from submodel 1 & 2
        final_mask = whole_tumour_mask * tumour_mask + (1 - whole_tumour_mask) * x

        return final_mask
