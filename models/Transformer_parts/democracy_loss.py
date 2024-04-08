import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class Democracy_loss(torch.nn.Module):
    """docstring for Democracy_loss."""

    def __init__(self, temperature, embed_dim=16):
        super(Democracy_loss, self).__init__()
        self.temperature = temperature
        self.base_temperature = 1.0 #set as 1 till I know what do to with it
        self.embed = nn.Sequential(nn.Linear(300*25*16,128), #implment a FCN that returns a vector of length embed_dim
                                    nn.ReLU(),
                                    nn.Linear(128,128)
                                    )

    def merge_all_examples_n_pad(self,*args):
        """This function puts all the examples (of different sizes) together
            and leaves zeros matrices in the empty slots.
        """
        num_examples, max_size, rest, arr, device = args
        final_arr = torch.zeros((num_examples, max_size, *rest)) #might need to move to device
        # This mask would come in handy to remove the effect of the zero matrix paddings
        mask = torch.zeros((num_examples,max_size))

        for i,x in enumerate(arr):
            final_arr[i,:len(x)] = x
            mask[i,:len(x)] = 1.

        return final_arr.to(device), mask.to(device)



    def get_hard_negs_n_positives(self,*args):
        """Returns hard negatives from the examples given. Negatives are determined using the policy,
            neg(Y) = X if top1(Y) is top2(X)
            In summary, anyone misclassified as me is a negative.
        """
        (examples_to_be_corrected, wrong_closest_pair,
                                    positive_samples_of) = args

        num_examples = len(examples_to_be_corrected)
        hard_negs = []
        pos_example = []
        sizes_neg = []
        sizes_pos = []

        top1 = wrong_closest_pair[:,0]
        top2 = wrong_closest_pair[:,1]

        for i in range(num_examples):
            hard_negs.append([]) #prepare a list to hold all negatives

            for j in range(num_examples):

                if top1[j] == top2[i]:
                    hard_negs[i].append(examples_to_be_corrected[j].unsqueeze(0))

            # positive_samples_of[str(top1[i])] is a negative example
            if len(hard_negs[i]) != 0:
                # This is when we are able to find hard negative examples

                # positive_samples_of[str(top1[i])]: #Note that some times this can be empty
                hard_negs[i] = torch.cat( ( positive_samples_of[str(top1[i])], # we join other negatives to the hard negatives
                                            torch.cat(hard_negs[i]) ) # we fuse all hard negatives together
                                        )

            else: #This is so we don't have empty negative examples. Though this maybe empty
                hard_negs[i] = positive_samples_of[str(top1[i])]

            sizes_neg.append( hard_negs[i].size(0) )

            pos_example.append(positive_samples_of[str(top2[i])])

            sizes_pos.append( pos_example[i].size(0) )

###########FIND A WAY TO DEAL WITH THESE EMPTY EXAMPLES
            # if pos_example[i].size(0) == 0:
            #     print(f"Example {i} has no positive example.")
            #     print(f"INFO: top1(W)={top1[i]}, top2(R)={top2[i]}, neg={sizes_neg[i]}, pos={sizes_pos[i]}, num_examples={num_examples}\n")
            # if hard_negs[i].size(0) == 0:
            #     print(f"Example {i} has no negative example.")
            #     print(f"INFO: top1(W)={top1[i]}, top2(R)={top2[i]}, neg={sizes_neg[i]}, pos={sizes_pos[i]}, num_examples={num_examples}\n")


        # find out shape of hard_negs: [num_examples x #number x time x n_nodes x filter_size]
        max_size_neg = max(sizes_neg) #maximum size of negatives
        max_size_pos = max(sizes_pos) #maximum size of positives
        _,*rest = examples_to_be_corrected.shape
        dev = examples_to_be_corrected.device #device needed

        result_neg, mask_neg = self.merge_all_examples_n_pad(num_examples, max_size_neg, rest, hard_negs, dev)
        # result_neg : [num_examples x max_size_neg x time x n_nodes x filter_size]
        result_pos, mask_pos = self.merge_all_examples_n_pad(num_examples, max_size_pos, rest, pos_example, dev)
        # result_pos : [num_examples x max_size_pos x time x n_nodes x filter_size]
        # mask_neg   : [num_examples x max_size] --tensor containing 1's and 0's
        return result_pos, mask_pos, result_neg, mask_neg


    def get_pos_n_neg_examples(self,*args):
        """
            So I am considering this: (note that top1(x) = label(x))
            let pos_ex_x be a  positive example of x, if top1(x) != label(x),
            but label(x) = top1(pos_ex_x), whether or not top2(pos_ex_x ) = top2(x).
            This should make the model robust to deal with other similar classes (say top3(x)).

            Then neg_ex_x is the negative example of x, if top2(x) = top1(neg_ex_x), whether or not top2(neg_ex_x) = top1(x).
        """
        # samples_of_further_pairs: [#further_pairs x time x n_nodes x filter_size]
        # samples_of_closest_pairs: [#closest_pairs x time x n_nodes x filter_size]
        # class_of_further_pair: [#further_pair x 2]
        # class_of_closest_pair: [#closest_pair x 2]
        # scores: [b x num_class]
        (label,
        samples_of_further_pairs, class_of_further_pair, idx_further_pair,
        samples_of_closest_pairs, class_of_closest_pair, idx_closest_pair) = args

        # I am only going to be considering cases where there are misclassification
        # likely due to the small margin between the top-2 scores.

        top1 = class_of_closest_pair[:,0]
        top2 = class_of_closest_pair[:,1]

        # if we consider top3, then we add (top3 == label[idx_closest_pair])
        required_condition = (top1 != label[idx_closest_pair]).logical_and(
                             (top2 == label[idx_closest_pair]))

        wrong_closest_pair = class_of_closest_pair[required_condition]
        # These conditions considers both a>w and w>a , since we are looking at the case where the true class is in top2.
        examples_to_be_corrected = samples_of_closest_pairs[required_condition] # get misclassified examples (these would form the anchors)
        # So far we have identified the classes to serve as anchors, but we have not explicitly identified the negative classes.
        # The negative classes are those classes that are close together, and w > a (instead of a > w).
        # Another way to formulate the negative class is this: for any pair of anchors (X, Y), X is a (hard-)negative of Y if top2(X)=top1(Y), and vice versa.
        # In summary, anyone misclassified as me is a negative.


        # We can also consider cases where the right class was given in top1, but they scores are still close together.
        # correct_closest_pair = class_of_closest_pair[(top1 == label[idx_closest_pair])]
        # correct_examples_to_be_pushed_apart = samples_of_closest_pairs[(top1 == label[idx_closest_pair])] #get correctly classified examples


        # Now we get the correctly classified examples that are further apart.
        # These would serve as the positive and negative examples.
        top1 = class_of_further_pair[:,0]
        required_condition = top1==label[idx_further_pair]
        correct_further_pair = class_of_further_pair[required_condition]
        correct_samples_of_further_pairs = samples_of_further_pairs[required_condition] #get correctly classified examples

        positive_samples_of = dict()
        # Note that for unique_class, we are considering all such (a,w) even when a<w
        unique_classes_of_closest_pair = wrong_closest_pair.flatten(0,-1).unique()
        for unique_class in unique_classes_of_closest_pair:
            # We need to be sure none of these positive samples are empty
            positive_samples_of[str(unique_class)] = correct_samples_of_further_pairs[correct_further_pair[:,0]==unique_class]

        positives, pos_mask , negatives, neg_mask = self.get_hard_negs_n_positives(examples_to_be_corrected,
                                                        wrong_closest_pair,
                                                        positive_samples_of)

        # examples_to_be_corrected : [#examples_to_be_corrected x time x n_nodes x filter_size]
        # negatives : [num_examples x max_size_neg x time x n_nodes x filter_size]
        # positives : [num_examples x max_size_pos x time x n_nodes x filter_size]
        # mask_neg   : [num_examples x max_size] --tensor containing 1's and 0's
        return examples_to_be_corrected, positives, pos_mask , negatives, neg_mask




    def signatures(self, *args):
        """
            We compute the signatures using the given arguments.
            Note that pos_n_neg_of is a function to obtain positive and negative samples.
        """
        # Currently thinking of using some sort of circle definition to keep all the
        # embeddings of the examples_to_be_corrected within a fixed radius that would
        # avoid as many negative_examples as possible (set by the radius--to be some hyperparameter).
        # I am also thinking of using some definition of the kernels from support vector machine.
        # Finally, I have decided to go with Supervised Contrastive loss:
        # https://arxiv.org/pdf/2004.11362.pdf (recommended by Yiming Ma)
        examples_to_be_corrected, positives, pos_mask , negatives, neg_mask = args
        # #examples_to_be_corrected = num_examples
        # examples_to_be_corrected : [num_examples x time x n_nodes x filter_size]
        # negatives : [num_examples x max_size_neg x time x n_nodes x filter_size]
        # positives : [num_examples x max_size_pos x time x n_nodes x filter_size]
        # mask_neg   : [num_examples x max_size] --tensor containing 1's and 0's


        pos = F.normalize(self.embed(positives.flatten(-3,-1)), dim=-2) # [num_examples x max_size_pos x embed_dim]
        pos[pos != pos] = 0. #This is to deal with cases where anchors don't have positive example(s)
                            #This also handles the cases where we have introduced paddings (as a result of merging the examples together)

        neg = F.normalize(self.embed(negatives.flatten(-3,-1)), dim=-2) # [num_examples x max_size_neg x embed_dim]
        neg[neg != neg] = 0. #This is to deal with cases where anchors don't have negative example(s)
                            #This also handles the cases where we have introduced paddings (as a result of merging the examples together)

        anchor = F.normalize(self.embed(examples_to_be_corrected.flatten(-3,-1)), dim=-2) # [num_examples x embed_dim]

        # We expand the dimension of the anchor to match the dimension of the negative_examples
        # then we transpose to make it match for matrix multiplication
        expnd_trans_anchor = anchor.unsqueeze(1).transpose(-2,-1) # [num_examples x embed_dim x 1]
        each_A_i = torch.div(neg.matmul(expnd_trans_anchor), self.temperature).squeeze(-1) # [num_examples x max_size_neg]
        each_A_i = each_A_i * neg_mask #multiply with mask to remove effect of those zero paddings
                                        #This mask may not be necessary anymore, as the effect of
                                        # neg[neg != neg] = 0. may have nullified it's effect.

        # Try to introduce into your code what author did for numerical stability
        each_A_i_max, _ = torch.max(each_A_i, dim=-1, keepdim=True)
        each_A_i_stab = each_A_i - each_A_i_max.detach() #This is for stability

        exp_each_A_i = torch.exp(each_A_i_stab) # [num_examples x max_size_neg]
        sum_over_exp_each_A_i = exp_each_A_i.sum(dim=-1) # [num_examples]
        log_sum_A_i = torch.log(sum_over_exp_each_A_i) # [num_examples]

        numerator = torch.div(pos.matmul(expnd_trans_anchor), self.temperature).squeeze(-1) # [num_examples x max_size_pos]

        log_prob = numerator - log_sum_A_i.unsqueeze(-1) # [num_examples x max_size_pos]
        # compute mean of log-likelihood over positive
        mean_log_prob_pos = log_prob.sum(dim=-1) / numerator.size(-1) # [num_examples]

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos # [num_examples]

        return loss.mean() # now a scalar



    def forward(self, *args):

        sig_args = self.get_pos_n_neg_examples(*args)
        loss = self.signatures(*sig_args)

        return loss



class XEnt_n_Democracy(torch.nn.Module):

    """This is a combination of the cross-entropy loss and the Democracy_loss"""

    def __init__(self, k1=1, temperature=0.1):
        super(XEnt_n_Democracy, self).__init__()
        self.k1 = k1 #constant to control the contribution of the Democracy_loss loss to the overall model
        self.democracy = Democracy_loss(temperature)
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.notify = 0 #This is to make us aware when democracy loss starts

    def forward(self, *args):

        scores, label, *democrats = args

        if democrats[0] is None:
            return self.cross_entropy(scores,label)
        else:
            self.notify += 1
            if self.notify == 1: #This is so that the prompt is printed only once
                print("Now Using Democracy...")
            args = (label, *democrats)
            return (
                    self.cross_entropy(scores,label)
                    + self.k1 * self.democracy(*args)
                    )
