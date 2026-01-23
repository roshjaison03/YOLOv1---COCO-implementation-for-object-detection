import torch
import torch.nn as nn

"""
YOLOv1 architecture with BatchNorm and debug statements.

Input  : (B, 3, 448, 448)
Output : (B, S*S*(C + B*5))
"""

# -------------------------------
# Architecture configuration
# -------------------------------
architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


# -------------------------------
# CNN Block
# -------------------------------
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, debug=False, **kwargs):
        super().__init__()
        self.debug = debug
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.1)

    def forward(self, x):
        if self.debug:
            print(f"[CNNBlock] input: {tuple(x.shape)}")

        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        if self.debug:
            print(f"[CNNBlock] output: {tuple(x.shape)}")

        return x


# -------------------------------
# YOLOv1 Model
# -------------------------------
class Yolov1(nn.Module):
    def __init__(
        self,
        in_channels=3,
        split_size=7,
        num_boxes=2,
        num_classes=1,
        debug=False,
    ):
        super().__init__()

        self.S = split_size
        self.B = num_boxes
        self.C = num_classes
        self.debug = debug

        self.darknet = self._create_conv_layers(architecture_config)
        self.fcs = self._create_fcs()

    # ---------------------------
    # Forward pass
    # ---------------------------
    def forward(self, x):
        if self.debug:
            print(f"\n[YOLO] Input image: {tuple(x.shape)}")

        x = self.darknet(x)

        if self.debug:
            print(f"[YOLO] After darknet: {tuple(x.shape)}")

        # YOLOv1 invariant
        assert x.shape[1:] == (1024, self.S, self.S), (
            f"Expected (1024, {self.S}, {self.S}), got {x.shape[1:]}"
        )

        x = torch.flatten(x, start_dim=1)

        if self.debug:
            print(f"[YOLO] After flatten: {tuple(x.shape)}")

        x = self.fcs(x)

        if self.debug:
            print(f"[YOLO] Final output: {tuple(x.shape)}")

        return x

    # ---------------------------
    # Build convolutional layers
    # ---------------------------
    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = 3

        for block in architecture:
            if isinstance(block, tuple):
                kernel, filters, stride, padding = block
                layers.append(
                    CNNBlock(
                        in_channels,
                        filters,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                        debug=self.debug,
                    )
                )
                in_channels = filters

            elif isinstance(block, str):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

                if self.debug:
                    layers.append(DebugLayer("MaxPool"))

            elif isinstance(block, list):
                conv1, conv2, repeats = block
                for _ in range(repeats):
                    layers.append(
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                            debug=self.debug,
                        )
                    )
                    layers.append(
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                            debug=self.debug,
                        )
                    )
                    in_channels = conv2[1]

        return nn.Sequential(*layers)

    # ---------------------------
    # Fully connected layers
    # ---------------------------
    def _create_fcs(self):
        output_dim = self.S * self.S * (self.C + self.B * 5)

        if self.debug:
            print(
                f"[YOLO] FC expects input: {1024*self.S*self.S}, "
                f"output: {output_dim}"
            )

        return nn.Sequential(
            nn.Linear(1024 * self.S * self.S, 4096),
            nn.Dropout(0.0),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, output_dim),
        )


# -------------------------------
# Debug helper layer
# -------------------------------
class DebugLayer(nn.Module):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def forward(self, x):
        print(f"[{self.name}] output: {tuple(x.shape)}")
        return x
