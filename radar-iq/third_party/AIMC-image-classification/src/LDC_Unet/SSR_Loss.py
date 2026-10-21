import torch
import torch.nn as nn

class SSRLoss(nn.Module):
    def __init__(self, num_classes, feature_dim, lambda_reg=0.12):
        super(SSRLoss, self).__init__()
        self.lambda_reg = lambda_reg  # Weight factor for L1 loss
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.centers = nn.Parameter(torch.randn(num_classes, feature_dim))  # Learnable class centers
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, features, labels, logits):
        """
        :param features: Feature vectors before FC layer (batch_size, feature_dim)
        :param labels: Class labels (batch_size,)
        :param logits: Output logits before softmax (batch_size, num_classes)
        """
        # Compute softmax loss
        loss_softmax = self.cross_entropy(logits, labels)

        # Get the class centers for each sample
        centers_batch = self.centers[labels]  # Shape: (batch_size, feature_dim)

        # Compute self-regularization loss (L1 distance)
        loss_reg = torch.mean(torch.abs(features - centers_batch))

        # Final SSR-Loss
        loss = loss_softmax + self.lambda_reg * loss_reg
        return loss