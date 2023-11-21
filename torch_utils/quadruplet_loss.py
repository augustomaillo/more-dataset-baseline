import torch
import torch.nn.functional as F


class QuadrupletLoss(torch.nn.Module):
    """
    Quadruplet loss function.
    Builds on the Triplet Loss and takes 4 data input: one anchor, one positive and two negative examples. The negative examples needs not to be matching the anchor, the positive and each other.
    """
    def __init__(self, margin1=2.0, margin2=1.0):
        super(QuadrupletLoss, self).__init__()
        self.margin1 = margin1
        self.margin2 = margin2

    def forward(self, anchor, positive, negative1, negative2):
        squarred_distance_pos = (anchor - positive).pow(2).sum(1)
        squarred_distance_neg = (anchor - negative1).pow(2).sum(1)
        squarred_distance_neg_b = (negative1 - negative2).pow(2).sum(1)

        quadruplet_loss = \
            F.relu(self.margin1 + squarred_distance_pos - squarred_distance_neg) \
            + F.relu(self.margin2 + squarred_distance_pos - squarred_distance_neg_b)

        return quadruplet_loss.mean()
    
class MyQuadrupletLoss(torch.nn.Module):
    def __init__(self, margin, device='cpu'):
        super(MyQuadrupletLoss, self).__init__()
        self.margin = margin
        self.zero = torch.zeros(1, device=device)
        
    def forward(self, camA_features, camA_labels, camB_features, camB_labels):
        """Computes quadruplet loss

        Args:
            camA_features (torch.Tensor): (n_samples, feature_dims)
            camA_labels (torch.tensor): (n_samples, n_classes)
            camB_features (torch.tensor): (n_samples, feature_dims)
            camB_labels (torch.tensor): (n_samples, n_classes)

        Returns:
            torch.Tensor: (1) Loss
        """
        with torch.no_grad():
            dist = torch.cdist(camA_features, camB_features) + 1e-6
            
            mask = torch.eq(camB_labels.argmax(dim=-1).unsqueeze(0).unsqueeze(-1), camA_labels.argmax(dim=-1).unsqueeze(0).unsqueeze(1)).float()            

            # # maior distancia entre pares positivos no batch
            dist_max = torch.mul(dist,mask).max(axis=-1)[0]

            # # menor distancia entre pares negativos no batch
            # # faz as distancias dos pares positivos serem iguais a maior distancia da matrix
            dist_min,dist_min_idx = torch.where(mask==1, dist.max(), dist).min(axis=-1)


            # # axis=0 -> min across columns 
            # #   (i.e for each anchor x and sample y, that y is the closest to x
            # #   retrieves z such that z is the closest to y )
            dist_negpair,_ = torch.min(
                dist[:,dist_min_idx].squeeze(1),axis=0)
            

            loss = torch.mean(torch.max(dist_max - dist_min + self.margin, self.zero)) + \
                torch.mean(torch.max(dist_max - dist_negpair + self.margin, self.zero))
            
        return loss