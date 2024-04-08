import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.metrics.confusion_matrix import ConfusionMatrix

class HCC(nn.Module):
    """This loss function is for Highly Correlated Classes (HCC)."""

    def __init__(self, num_class, k_corr=20):
        """
            k_corr: number of correlated classes we want
        """
        super(HCC, self).__init__()

        self.num_class = num_class
        self.num_class_we_want = k_corr


    def rearrange_max_to_last(self, masked_cm):
        """
            This returns a sorted tensor of both the entries and index
        """

        #[belong to: idx, predicted: default index/classes, amount: entry]
        entry , idx = torch.max(masked_cm , dim=0) # Returns maximum across each column
        classes = torch.arange(self.num_class)

        # Find the classes that appears most (has highest count)
        unique_idx, num_of_each_class = idx.unique(return_counts=True)

        # Rank classes based on number of "correlation"
        ent , idx_ent = torch.sort(num_of_each_class, descending=True)
        tmp = unique_idx + 0 #To ensure it is deep copied
        for i in range(len(idx_ent)):
            unique_idx[i] = tmp[idx_ent[i]]
            # unique_idx : classes in order of descending correlation

        # this is a list of classes that are highly corr to our ranked classes
        mem_of_each_class = []
        num_of_hcSamples_of_each_class = []
        for value in unique_idx:
            # Get the index of those classes in idx
            locations = torch.where(idx==value)[0].numpy()
            mem_of_each_class.append(locations)

            total = 0
            for each in locations:
                total += entry[each] # get the number of samples correlated to our ranked class

            num_of_hcSamples_of_each_class.append(total)

        # Now we rank according to the total number of samples
        num_of_hcSamples_of_each_class = torch.tensor(num_of_hcSamples_of_each_class)
        num_hcSamples, idx_num_hcSamples = torch.sort(num_of_hcSamples_of_each_class, descending=True)

        tmp_unique_idx = unique_idx + 0 #To ensure it is deep copied
        tmp_mem_of_each_class = mem_of_each_class

        for i in range(len(unique_idx)):
            unique_idx[i] = tmp_unique_idx[idx_num_hcSamples[i]]
            # mem_of_each_class[i] = tmp_mem_of_each_class[idx_num_hcSamples[i]]
            # unique_idx : classes in order of descending correlation (by number of samples)


        return unique_idx[:self.num_class_we_want].numpy()



    def confusion_matrix(self, targets, scores):

        predictions = scores.max(dim=1)[1] #Returns the indices of the maximum value

        # initialize confusion matrix
        confusion = torch.zeros(self.num_class, self.num_class)

        # flatten inputs
        predictions = predictions.view(-1)
        targets = targets.view(-1)

        # fill in confusion matrix
        for p, t in zip(targets,predictions):
            confusion[p.long(), t.long()] += 1

        return confusion



    def get_HCC(self, y, scores):
        """
            This function return a list of the highly correlated classes
            y       : batch_size x 1
            y_hat   : batch_size x num_class
        """
        mask = torch.ones(self.num_class, self.num_class) - torch.eye(self.num_class)
        cm = self.confusion_matrix(y, scores)
        out = cm * mask
        ranked_corr_classes = self.rearrange_max_to_last(out)

        return ranked_corr_classes


    def get_pairs_for_MMD(self, ranked_corr_classes):
        """Here we return the pairs for which we would compute their MMD^2"""

        k = self.num_class_we_want
        classes = list(range(self.num_class))
        hold_pairs = []
        for i,each in enumerate(ranked_corr_classes):
            num_mmd_pairs = k- i - 1
            mmd_pairs = [each]*num_mmd_pairs
            classes.remove(each) #We do not wish to compute the MMD of a class with itself

            arr = list(zip(mmd_pairs,classes))
            hold_pairs.append(arr)

        pairs_of_mmd = list(itertools.chain(*hold_pairs))

        return pairs_of_mmd


    def forward(self, y, scores):

        ranked_corr_classes = self.get_HCC(y, scores)
        pairs_of_mmd = self.get_pairs_for_MMD(ranked_corr_classes)

        return pairs_of_mmd
