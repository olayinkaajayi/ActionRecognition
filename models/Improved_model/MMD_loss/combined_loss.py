import torch
import torch.nn as nn
import torch.nn.functional as F

from MMD_loss.sparse_mmd import SparseMMD

class Xent_n_SparseMMD(nn.Module):
    """
        The Xent_n_SparseMMD module combines the corss-entropy loss with
        the tailor made SparseMMD loss.
    """

    def __init__(self, num_class, num_corr_clas_needed=20, ls=0.2, lk=1.0):
        super(Xent_n_SparseMMD, self).__init__()

        self.Xent = nn.CrossEntropyLoss(label_smoothing=ls)
        self.sparseMMD = SparseMMD(num_class, num_corr_clas_needed)
        self.lk = lk # hyperparameter to determine the contribution of the SparseMMD loss

    def forward(self, *args):
        """
            sample_data : batch_size x hidden_size
            scores      : the raw scores for each class from our model (batch_size)
            y_gt        : ground truth class label (batch_size)
        """
        epoch, scores, y_gt, sample_data = args

        return (
                self.Xent(scores, y_gt) + #CrossEntropyLoss
                self.lk*self.sparseMMD(sample_data, y_gt, scores, epoch=epoch) #Mod-MMD_loss
                )
