import torch
import torch.nn as nn
import torch.nn.functional as F

from MMD_loss.mmd_loss import ModMMD
from MMD_loss.hcc_loss import HCC

class SparseMMD(nn.Module):
    """
        This implements a sparse sum of the MMD for each
        pair of selected classes.
    """

    def __init__(self, num_class, num_corr_clas_needed=20):
        super(SparseMMD, self).__init__()

        self.get_hcc = HCC(num_class, k_corr=num_corr_clas_needed)
        self.mmd = ModMMD()
        self.k_corr_sample_size = (num_class*num_corr_clas_needed +
                                    num_corr_clas_needed*(num_corr_clas_needed-1)/2.0)

        self.pairs_of_mmd = []

    def data_seperated_to_classes(self, y_gt, epoch):

        idx_data_per_class = {}
        for i,y in enumerate(y_gt): # identifies the index where each class exists
            try:
                idx_data_per_class[y.item()].append(i)
            except KeyError:
                idx_data_per_class[y.item()] = []
                idx_data_per_class[y.item()].append(i)

        return idx_data_per_class


    def compute_mmd(self, sample_data, data_per_class, pairs_of_mmd):

        total_mmd = 0
        num_of_pairs_available = 0
        for each in pairs_of_mmd:

            # Each mini batch may not contain certain class samples we may need,
            # so we skip those computations in the meantime
            if (each[0] in data_per_class.keys()) and (each[1] in data_per_class.keys()):

                idx_x , idx_y = data_per_class[each[0]] , data_per_class[each[1]]
                x , y = sample_data[idx_x], sample_data[idx_y]
                total_mmd += self.mmd(x, y)
                # We count how many computations were available as not all classes
                # are represented in each mini batch.
                # So we use num_of_pairs_available in place of self.k_corr_sample_size
                num_of_pairs_available += 1

        return total_mmd/((num_of_pairs_available*1.0) if num_of_pairs_available != 0 else 1)


    def forward(self, sample_data, y_gt, scores, epoch=0):
        """
            sample_data : batch_size x hidden_size
            y_gt        : ground truth class label (batch_size)
            y_pred      : predicted label (batch_size) # check this dimension because of ignite.metric
        """
        # We want to check for the HCCs only at every 5 epochs.
        # We assume it would not change much within this time.
        # This would also make our computation faster.
        if (epoch%5) == 0:
            self.pairs_of_mmd = self.get_hcc(y_gt, scores)

        # This is a dictionary for each class and the index of the
        # samples belonging to it.
        data_per_class = self.data_seperated_to_classes(y_gt, epoch)
        loss = self.compute_mmd(sample_data, data_per_class, self.pairs_of_mmd)

        return loss
