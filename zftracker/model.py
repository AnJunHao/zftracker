import torch.nn
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from icecream import ic
import torch

from .util.tqdm import TQDM as tqdm

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

class GeometricDense(torch.nn.Module):

    def __init__(self,
                 input_size=180,
                 num_dense_layers=1,
                 dense_units=128,
                 dropout=0.5):
        
        super().__init__()
        preprocess_layers = [
            torch.nn.Linear(input_size, dense_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        ]
        for _ in range(num_dense_layers - 1):
            preprocess_layers.extend([
                torch.nn.Linear(dense_units, dense_units),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ])
        self.preprocess = torch.nn.Sequential(*preprocess_layers)
    
        # HMT Output Layer
        self.hmt_output = self.get_dense_layers(
            dense_units, 6, num_dense_layers, dense_units, dropout)

        # HM Output Layer
        self.hm_output = self.get_dense_layers(
            dense_units+2, 4, num_dense_layers, dense_units, dropout)

        # HT Output Layer
        self.ht_output = self.get_dense_layers(
            dense_units+2, 4, num_dense_layers, dense_units, dropout)

        # MT Output Layer
        self.mt_output = self.get_dense_layers(
            dense_units+2, 4, num_dense_layers, dense_units, dropout)
        
        # H Output Layer
        self.h_output = self.get_dense_layers(
            dense_units+4, 2, num_dense_layers, dense_units, dropout)
        
        # M Output Layer
        self.m_output = self.get_dense_layers(
            dense_units+4, 2, num_dense_layers, dense_units, dropout)
        
        # T Output Layer
        self.t_output = self.get_dense_layers(
            dense_units+4, 2, num_dense_layers, dense_units, dropout)
        
        self.output_layers = {
            'hmt': self.hmt_output,
            'hm': self.hm_output,
            'ht': self.ht_output,
            'mt': self.mt_output,
            'h': self.h_output,
            'm': self.m_output,
            't': self.t_output
        }
        
    def forward(self, x: dict):

        seq_x = x['seq']
        seq_x = self.preprocess(seq_x)

        output = {}

        for key in x:
            if key == 'seq':
                continue
            coords_x = x[key]
            coords_x = torch.cat([seq_x, coords_x], dim=1)
            coords_x = self.output_layers[key](coords_x)
            output[key] = coords_x
        
        return output

    def get_dense_layers(self, input_size, output_size, num_dense_layers, dense_units, dropout):
        layers = [
            torch.nn.Linear(input_size, dense_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        ]
        for _ in range(num_dense_layers - 1):
            layers.extend([
                torch.nn.Linear(dense_units, dense_units),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ])
        layers.extend([
            torch.nn.Linear(dense_units, output_size),
        ])
        return torch.nn.Sequential(*layers)

class ResNetHeatmapClassify(torch.nn.Module):
    
    def __init__(self,
                 resnet="resnet50",
                 dropout=0.5,
                 apply_sigmoid=True,
                 pretrained=False):
        super(ResNetHeatmapClassify, self).__init__()
        
        # Load a pre-trained ResNet model
        resnets = {
            "resnet18": (resnet18, 512),
            "resnet34": (resnet34, 512),
            "resnet50": (resnet50, 2048),
            "resnet101": (resnet101, 2048),
            "resnet152": (resnet152, 2048),
        }
        model_creator, num_features = resnets[resnet]
        self.resnet = model_creator(pretrained=pretrained)
        
        # Modify the first convolutional layer to accept 5-channel input
        self.resnet.conv1 = torch.nn.Conv2d(5, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Add a dropout layer
        self.dropout = torch.nn.Dropout(dropout)
        
        # Replace the fully connected layer of the ResNet model to output a single value
        self.resnet.fc = torch.nn.Linear(num_features, 1)
        
        # Apply sigmoid to the output if specified
        self.apply_sigmoid = apply_sigmoid
        
    def forward(self, x):
        # Pass the input through the ResNet model
        x = self.resnet(x)
        
        # Apply dropout to the penultimate feature vector
        x = self.dropout(x)
        
        # Apply sigmoid if needed
        if self.apply_sigmoid:
            x = torch.sigmoid(x)
        
        # Return the final output
        return x


class ResNetVectorClassify(torch.nn.Module):
    def __init__(self,
                 resnet="resnet50",
                 vector_expand=128,
                 dropout=0.5,
                 num_dense_layers=0,
                 dense_units=512,
                 apply_sigmoid=True,
                 pretrained=False):
        super(ResNetVectorClassify, self).__init__()
        
        # Load a pre-trained ResNet model
        # Define a dictionary to map resnet name to its corresponding model
        resnets = {
            "resnet18": (resnet18, 512),
            "resnet34": (resnet34, 512),
            "resnet50": (resnet50, 2048),
            "resnet101": (resnet101, 2048),
            "resnet152": (resnet152, 2048),
        }

        self.resnet = resnets[resnet][0](pretrained=pretrained)
        
        # Remove the last layer (fully connected layer) of the ResNet model
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])

        # We need to forward a dummy variable through the ResNet's children
        # to get the number of features output by the conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 256, 256)
            resnet_features = self.resnet(dummy_input).shape[1]
        
        # Add a fully connected layer for the 12D vector
        self.vector_fc = torch.nn.Linear(12, vector_expand)

        if dropout:
            self.vector_fc = torch.nn.Sequential(
                self.vector_fc,
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout)
            )
        else:
            self.vector_fc = torch.nn.Sequential(
                self.vector_fc,
                torch.nn.ReLU()
            )
        
        # Combine features from the image and the vector
        layers = [
            torch.nn.Linear(resnet_features + vector_expand, dense_units)]
        if dropout:
            for _ in range(num_dense_layers):
                layers.extend([
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout),
                    torch.nn.Linear(dense_units, dense_units)
                ])
            layers.extend([
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(dense_units, 1)
            ])
        else:
            for _ in range(num_dense_layers):
                layers.extend([
                    torch.nn.ReLU(),
                    torch.nn.Linear(dense_units, dense_units)
                ])
            layers.extend([
                torch.nn.ReLU(),
                torch.nn.Linear(dense_units, 1)
            ])
        if apply_sigmoid:
            layers.append(torch.nn.Sigmoid())
        self.dense_layers = torch.nn.Sequential(*layers)
        
    def forward(self, x):
        # Unpack the input
        image, vector = x

        # Forward pass through the ResNet model
        image_features = self.resnet(image)
        
        # Flatten the image features
        image_features = image_features.view(image_features.size(0), -1)
        
        # Forward pass through the vector fully connected layer
        vector_features = self.vector_fc(vector)
        
        # Concatenate the image and vector features
        combined_features = torch.cat((image_features, vector_features), dim=1)
        
        # Forward pass through the combined fully connected layers
        result = self.dense_layers(combined_features)
        
        return result
    
class ResNetSimpleClassify(torch.nn.Module):
    def __init__(
        self,
        resnet="resnet50",
        pretrained=False,
        dropout=0.5,
        apply_sigmoid=True,
        num_dense_layers=0,
        dense_units=512,
        freeze_resnet=False,
    ):
        super(ResNetSimpleClassify, self).__init__()

        # Load a pre-trained ResNet model
        resnets = {
            "resnet18": (resnet18, 512),
            "resnet34": (resnet34, 512),
            "resnet50": (resnet50, 2048),
            "resnet101": (resnet101, 2048),
            "resnet152": (resnet152, 2048),
        }

        self.resnet = resnets[resnet][0](pretrained=pretrained)
        
        # Remove the last layer (fully connected layer) of the ResNet model
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-1])

        # We need to forward a dummy variable through the ResNet's children
        # to get the number of features output by the conv layers
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 160, 160)
            resnet_features = self.resnet(dummy_input).shape[1]

        # Freeze the ResNet model if specified
        if freeze_resnet:
            for param in self.resnet.parameters():
                param.requires_grad = False

        # Define a fully connected layer to classify the input
        dense_layers = [
            torch.nn.Linear(resnet_features, dense_units),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        ]
        for _ in range(num_dense_layers):
            dense_layers.extend([
                torch.nn.Linear(dense_units, dense_units),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ])
        dense_layers.extend([
            torch.nn.Linear(dense_units, 1),
        ])

        self.dense = torch.nn.Sequential(*dense_layers)

        # Apply sigmoid to the output if specified
        self.apply_sigmoid = apply_sigmoid

    def unfreeze(self):
        for param in self.resnet.parameters():
            param.requires_grad = True

    def forward(self, x):
        # Forward pass through the ResNet model
        x = self.resnet(x)

        # Flatten the output from the ResNet model
        x = x.view(x.size(0), -1)

        # Forward pass through the fully connected layer
        x = self.dense(x)

        # Apply sigmoid if needed
        if self.apply_sigmoid:
            x = torch.sigmoid(x)

        return x

class ResNetDeconv(torch.nn.Module):
    def __init__(
        self,
        num_keypoints,
        num_deconv=3,
        resnet="resnet50",
        deconv_channels=256,
        pretrained=False,
        dropout=False,
        sigmoid=True,
        padding=1,
        output_padding=0,
    ):
        super(ResNetDeconv, self).__init__()

        # Define a dictionary to map resnet name to its corresponding model
        resnets = {
            "resnet18": (resnet18, 512),
            "resnet34": (resnet34, 512),
            "resnet50": (resnet50, 2048),
            "resnet101": (resnet101, 2048),
            "resnet152": (resnet152, 2048),
        }

        # Get the resnet model from the dictionary
        self.resnet = resnets[resnet][0](pretrained=pretrained)

        # Remove the last layer (fc layer) from ResNet
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-2])

        # Get the number of output channels from the resnet
        # I don't know how to get the output channels
        # So I hard coded the channels manually
        num_out_channels = resnets[resnet][1]

        # Add deconvolutional layers with batch normalization and ReLU activation
        layers = []
        if isinstance(padding, int):
            padding = (padding,) * num_deconv
        elif isinstance(padding, tuple) or isinstance(padding, list):
            if len(padding) != num_deconv:
                raise ValueError(
                    f"Argument 'padding' should have the same lenth as 'num_deconv', received {padding}"
                )
        else:
            raise TypeError(
                f"Argument 'padding' should be int or tuple/list, received {padding}"
            )
        if isinstance(output_padding, int):
            output_padding = (output_padding,) * num_deconv
        elif isinstance(output_padding, tuple) or isinstance(output_padding, list):
            if len(output_padding) != num_deconv:
                raise ValueError(
                    f"Argument 'output_padding' should have the same lenth as 'num_deconv', received {output_padding}"
                )
        else:
            raise TypeError(
                f"Argument 'output_padding' should be int or tuple/list, received {output_padding}"
            )

        for i in range(num_deconv):
            in_channels = num_out_channels if i == 0 else deconv_channels
            layers += [
                torch.nn.ConvTranspose2d(
                    in_channels,
                    deconv_channels,
                    kernel_size=4,
                    stride=2,
                    padding=padding[i],
                    output_padding=output_padding[i],
                ),
                torch.nn.BatchNorm2d(deconv_channels),
                torch.nn.ReLU(inplace=True),
            ]
        self.deconvs = torch.nn.Sequential(*layers)

        # Add dropout if specified
        self.dropout = dropout
        if self.dropout:
            self.dropout_layer = torch.nn.Dropout(self.dropout)

        # Add a 1x1 convolutional layer to generate the predicted heatmaps
        self.prediction = torch.nn.Conv2d(
            deconv_channels, num_keypoints, kernel_size=1)

        # Apply sigmoid if required
        self.sigmoid = sigmoid
        if self.sigmoid:
            self.sigmoid_layer = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.deconvs(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.prediction(x)
        if self.sigmoid:
            x = self.sigmoid_layer(x)
        return x


class ResnetOffsetDeconv(ResNetDeconv):
    def __init__(
        self,
        num_keypoints,
        num_deconv=3,
        resnet="resnet50",
        deconv_channels=256,
        pretrained=False,
        dropout=False,
        sigmoid=True,
        padding=1,
        output_padding=0,
    ):
        super().__init__(
            num_keypoints * 3,
            num_deconv,
            resnet,
            deconv_channels,
            pretrained,
            dropout,
            sigmoid,
            padding,
            output_padding,
        )
        self.num_keypoints = num_keypoints

    def forward(self, x):
        x = self.resnet(x)
        x = self.deconvs(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.prediction(x)
        if self.sigmoid:
            # Apply sigmoid to the [0, 3, 6, ..., 3*(num_keypoints-1)] channels of x
            x[:, [i * 3 for i in range(self.num_keypoints)]] = self.sigmoid_layer(
                x[:, [i * 3 for i in range(self.num_keypoints)]]
            )

        return x

class ResnetOffsetDoublecheckDeconv(ResNetDeconv):

    def __init__(
        self,
        num_keypoints,
        num_deconv=3,
        resnet="resnet50",
        deconv_channels=256,
        pretrained=False,
        dropout=False,
        sigmoid=True,
        padding=1,
        output_padding=0,
    ):
        super().__init__(
            num_keypoints * 5,
            num_deconv,
            resnet,
            deconv_channels,
            pretrained,
            dropout,
            sigmoid,
            padding,
            output_padding,
        )
        self.num_keypoints = num_keypoints

    def forward(self, x):
        x = self.resnet(x)
        x = self.deconvs(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.prediction(x)
        if self.sigmoid:
            # Apply sigmoid to the [0, 5, 10, ..., 5*(num_keypoints-1)] channels of x
            x[:, [i * 5 for i in range(self.num_keypoints)]] = self.sigmoid_layer(
                x[:, [i * 5 for i in range(self.num_keypoints)]]
            )

        return x

class ResnetTripleOffsetDoublecheckClassifyDeconv(ResNetDeconv):

    def __init__(
        self,
        num_deconv=3,
        resnet="resnet50",
        deconv_channels=256,
        pretrained=False,
        dropout=False,
        sigmoid=True,
        padding=1,
        output_padding=0,
    ):
        super().__init__(
            27,
            num_deconv,
            resnet,
            deconv_channels,
            pretrained,
            dropout,
            sigmoid,
            padding,
            output_padding,
        )
        self.sigmoid_layer_classify = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.resnet(x)
        x = self.deconvs(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.prediction(x)
        if self.sigmoid:
            # Apply sigmoid to the [0, 7, 8, 9, 16, 17, 18, 25, 26] channels of x
            layers = [0, 9, 18]
            x[:, layers] = self.sigmoid_layer(
                x[:, layers]
            )
            classify_layers = [7, 8, 16, 17, 25, 26]
            x[:, classify_layers] = self.sigmoid_layer_classify(
                x[:, classify_layers]
            )

        return x

class ResnetTripleOffsetDoublecheckClassifyDeconvV2(ResNetDeconv):

    def __init__(
        self,
        num_deconv=3,
        resnet="resnet50",
        deconv_channels=256,
        pretrained=False,
        dropout=False,
        sigmoid=True,
        padding=1,
        output_padding=0,
    ):
        super().__init__(
            30, # For each keypoint class, we have 4 channels for classified prediction and 6 channels for regression
            num_deconv,
            resnet,
            deconv_channels,
            pretrained,
            dropout,
            sigmoid,
            padding,
            output_padding,
        )
    
    def forward(self, x):
        x = self.resnet(x)
        x = self.deconvs(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.prediction(x)
        if self.sigmoid:
            layers = [0, 1, 2, 3, 10, 11, 12, 13, 20, 21, 22, 23]
            x[:, layers] = self.sigmoid_layer(
                x[:, layers]
            )

        return x
    
class ResnetTripleOffsetDoublecheckDeconv(ResNetDeconv):
    def __init__(
        self,
        num_deconv=3,
        resnet="resnet50",
        deconv_channels=256,
        pretrained=False,
        dropout=False,
        sigmoid=True,
        padding=1,
        output_padding=0,
    ):
        super().__init__(
            21,
            num_deconv,
            resnet,
            deconv_channels,
            pretrained,
            dropout,
            sigmoid,
            padding,
            output_padding,
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.deconvs(x)
        if self.dropout:
            x = self.dropout_layer(x)
        x = self.prediction(x)
        if self.sigmoid:
            layers = [0, 7, 14]
            x[:, layers] = self.sigmoid_layer(
                x[:, layers]
            )

        return x