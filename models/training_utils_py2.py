import pickle
import csv
import os
import random

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from tqdm import tqdm
# from thop import profile, clever_format
# import deepspeed
# from deepspeed.profiling import flops_profiler
from deepspeed.profiling.flops_profiler import FlopsProfiler
from deepspeed.profiling.flops_profiler.profiler import get_model_profile

from models.feeder_ntu import Feeder

# def calculate_flops(model, A, bs=8, T=64, M=2, V=25, C=3, sample_data=None):
#     input_size = (bs, T, M*V, C)  # Change the input size according to your model's input shape
#     input_data = torch.randn(*input_size) if sample_data is None else sample_data
#
#     flops_fwd, params_fwd = profile(model, inputs=(input_data,A), verbose=False)
#
#     # Enable gradients for backward pass
#     input_data.requires_grad_()
#     model.zero_grad()
#
#     output = model(input_data, A)
#     grad_output = torch.randn_like(output)
#
#     flops_bwd, _ = profile(model, inputs=(input_data,A), verbose=False)
#
#     # flops_fwd, flops_bwd = clever_format([flops_fwd, flops_bwd], "%.2f")
#     return flops_fwd, flops_bwd



def calculate_flops(model, A, bs=8, T=64, M=2, V=25, C=3):
    input_size = (bs, T, M*V, C)  # Change the input size according to your model's input shape
    flops, macs, params = get_model_profile(model, input_shape=input_size, as_string=False,
                    kwargs={'A':A}, warm_up=5, output_file='saved_images/flop_cnt.txt')
    return flops


# def calculate_flops(model):
#   """Calculates the FLOPS of a PyTorch model using DeepSpeed Flops Profiler.
#
#   Args:
#     model: The PyTorch model to calculate the FLOPS for.
#
#   Returns:
#     The FLOPS of the model.
#   """
#
#   # Initialize the DeepSpeed Flops Profiler.
#   profiler = flops_profiler.FlopsProfiler(model)
#
#   # Profile the forward pass of the model.
#   profiler.profile()
#
#   # Get the total FLOPS of the forward pass.
#   forward_flops = profiler.get_total_flops()
#
#   # Profile the backward pass of the model.
#   profiler.profile(backwards=True)
#
#   # Get the total FLOPS of the backward pass.
#   backward_flops = profiler.get_total_flops()
#
#   # Return the total FLOPS of the model.
#   return forward_flops , backward_flops


# def calculate_flops(model, A, bs=8, T=64, M=2, V=25, C=3, sample_data=None):
#     # Set up DeepSpeed Flops Profiler
#     profiler = flops_profiler.FlopsProfiler(model)
#
#     # Create a sample input tensor with the desired input size
#     input_size = (bs, T, M*V, C)
#     input_tensor = torch.randn(*input_size) if sample_data is None else sample_data
#
#     # Enable the Flops Profiler
#     profiler.start_profile()
#
#     # Forward pass through the model
#     output = model(input_tensor,A)
#
#     # Backward pass through the model
#     output.sum().backward()
#
#     # Disable the Flops Profiler
#     profiler.end_profile()
#
#     # Calculate and return the FLOPS for both forward and backward pass
#     forward_flops = profiler.get_total_flops(enable_backward=False)
#     backward_flops = profiler.get_total_flops(enable_forward=False)
#
#     return forward_flops, backward_flops




def save_float_to_csv(value, filename):
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([value])


def read_csv(filename):
    data = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(float(row[0]))
    return data



data_path_fn = lambda data_type,num_class: f'/dcs/large/u2034358/ntu{num_class}/{data_type}_aligned.npz'


def repeat_skeleton_for_one_body(data, labels):
    """
        Our model requires that for actions involving one human,
        the second skeleton should be the same as the first skeleton.
        This is needed for the Interact module in our model.
    """
    for i,label in enumerate(labels):
        # We do this for the action class involving just one human
        if label < 49: #49 because label starts from 0
            data[i,:,1,:,:] = data[i,:,0,:,:]

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_data_infogcn(arg, window_size=64, xV_xS='NTU60_CV', num_class=60, repeat=True):
    # Feeder = import_class(arg.feeder)
    data_loader = dict()
    num_class = '' if num_class==60 else num_class
    data_path = data_path_fn(xV_xS, num_class)

    dt = Feeder(data_path=data_path,
        split='train',
        window_size=window_size,
        p_interval=[0.5, 1],
        random_rot= arg.random_rot,
        sort=True if arg.balanced_sampling else False,
        repeat=repeat
    )
    if arg.balanced_sampling:
        sampler = BS(data_source=dt, args=arg)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    data_loader['train'] = torch.utils.data.DataLoader(
        dataset=dt,
        sampler=sampler,
        batch_size=arg.batch_size,
        shuffle=shuffle,
        num_workers=arg.num_worker,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=init_seed)

    data_loader['test'] = torch.utils.data.DataLoader(
        dataset=Feeder(
            data_path=data_path,
            split='test',
            window_size=window_size,
            p_interval=[0.95]
        ),
        batch_size=arg.batch_size,
        shuffle=False,
        num_workers=arg.num_worker,
        drop_last=False,
        pin_memory=True,
        worker_init_fn=init_seed)

    return data_loader['train'], data_loader['test']

def load_data_align(split='train', xV_xS='NTU60_CV'):
        # data: N T M V C
        data_path = data_path_fn(xV_xS)
        npz_data = np.load(data_path)
        if split == 'train':
            data = npz_data['x_train']
            label = np.argmax(npz_data['y_train'], axis=-1)
        elif split == 'test':
            data = npz_data['x_test']
            label = np.argmax(npz_data['y_test'], axis=-1)
        else:
            raise NotImplementedError('data split only supports train/test')

        nan_out = np.isnan(data.mean(-1).mean(-1))==False
        data = data[nan_out]
        label = label[nan_out]
        # Now reshape data
        N, T, _ = data.shape
        data = data.reshape((N, T, 2, 25, 3))
        repeat_skeleton_for_one_body(data, label)

        return data.reshape(N,T,-1,3), label #Reshape: this way our model is prepared to receive it


def standardize(x,mean=None,std=None):
    """
        This function is meant to standardize the data.
        ie. mean 0 and std 1

        or for norm,
        the function scales the data to the range [tmin,tmax]
    """
    if (mean is None) and (std is None):
        mean = np.mean(x,axis=0)
        std = np.std(x,axis=0)

    return (x - mean)/std



def num_of_param(params):
    """Estimate number of parameters."""
    total = 0
    cnt = 0
    for p in params:
        if p.requires_grad:
            try:
                total += p.numel()
            except ValueError:
                cnt += 1
                print("caught an error {}".format(cnt))
                continue
    return total


def use_cuda(use_cpu,many=False,verbose=True):
    """
        Imports GPU if available, else imports CPU.
    """
    if use_cpu: #overrides GPU and use CPU
        device = torch.device("cpu")
        USE_CUDA = False
        if verbose:
            print("CPU used")
        return device,USE_CUDA

    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        device = torch.device("cuda" if many else "cuda:0")
        if verbose:
            print("GPU is available")
    else:
        device = torch.device("cpu")
        if verbose:
            print("GPU not available, CPU used")
    return device,USE_CUDA


def toTensor(v,device,dtype = torch.float,requires_grad = False):#This function converts the entry 'v' to a tensor data type
    return torch.tensor(v).type(dtype).to(device)


def toNumpy(v,device): #This function converts the entry 'v' back to a numpy (float) data type
    if device.type == 'cuda':
        return v.detach().cpu().numpy()
    return v.detach().numpy()


def moving_averages(arr, window_size=3):
    ''' Program to calculate moving average using numpy.
        Note: Choosing a fixed length (of 100) for the time series data,
        forces the window size to be quite large (50) even for time lengths
        a bit above 100. This may cause us to lose the fine-grain information
        at each step even though we may be erradicating noise from the data.

        So, to manage the information lose and reduce noise, we rather use a
        fixed (small) window size, and drop out (at random or using certain
        conditions) some of the time series data points to achieve the desired
        time length. Or we can employ the 1D CNN here.
    '''
    # We may consider using a 1D CNN in place of a moving average.
    # This is because we do not know exactly how much weight is associated
    # with each entry in the "window" that we average into the new time length.
    # So in place or an average, a maximum or any other metric, we rather use a
    # learned weigthed sum, which is a 1D CNN.

    # size of moving average is len(arr) - window_size + 1
    # window_size = len(arr) - new_time_len + 1

    i = 0
    # Initialize an empty list to store moving averages
    moving_averages = []

    # Loop through the array t o
    #consider every window of size 3
    while i < len(arr) - window_size + 1:

        # Calculate the average of current window
        window_average = np.sum(arr[i:i+window_size],axis=0) / window_size

        # Store the average of current
        # window in moving average list
        moving_averages.append(window_average)

        # Shift window to right by one position
        i += 1

    return np.array(moving_averages)



def pad_data(dat,max_time=300,stgcn=False, screen= False, norm=False, window_size=3, repeat=True):
    """
        dat is a dictionary.
        This function pads the time dimension of dat with a 25 x 3 zero matrix.
    """
    if not repeat:
        print("Using Zero padding!")

    if stgcn:
        dat_holder_across_time = []
        for each in dat:
            dat_time,row,col = each['skel_body0'].shape
            body_count = 2
            new_array = np.zeros((max_time,body_count,row,col))
            new_array[:dat_time,0] = each['skel_body0']
            each['skel_body0'] = new_array

            dat_holder_across_time.append(each['skel_body0'])
        return dat_holder_across_time

############### IF NOT STGCN###########
    dat_holder_across_time = []
    for cnt,each in enumerate(dat):
        result1 = each['skel_body0']
        # Remove noise
        if screen: # consider window_size=7
            if cnt == 0 :
                max_time = max_time - window_size + 1
            result1 = data_screening(result1, window_size=window_size)

        if result1.shape[0] < max_time:
            dat_time,row,col = result1.shape
            new_array = np.zeros((max_time,row,col))
            new_array[:dat_time] = result1
            result1 = new_array

        if 'skel_body1' in each.keys():
            result2 = each['skel_body1']
            # Remove noise
            if screen: # consider window_size=7
                if cnt == 0 :
                    max_time = max_time - window_size + 1
                result2 = data_screening(result2, window_size=window_size)

            if result2.shape[0] < max_time:
                dat_time,row,col = result2.shape
                new_array = np.zeros((max_time,row,col))
                new_array[:dat_time] = result2
                result2 = new_array
        else:
            if repeat:
                result2 = result1
            else:
                result2 = np.zeros((max_time,row,col))

        if norm:
            result1 = normalization_in_space(result1)
            result2 = normalization_in_space(result2)

        dat_holder_across_time.append(np.concatenate([result1, result2],axis=-2))

    return dat_holder_across_time


def normalization_in_space(data, kinetics= False):
    """We apply a normalization in space as described in the AMAB paper."""
    if kinetics:
        print("Normalizing data...")
        root_idx = [0,18]
        b,t,Np,d = data.shape
        root = np.zeros((b,t,1,d))
        root[:,:,:,[0,1]] = np.expand_dims(data[:,:,root_idx,[0,1]], axis=-2)
        return data - root
    else:
        root_idx = 0
        root = np.expand_dims(data[:,root_idx], axis=1)

    return data - root


def data_screening(data, window_size=3):
    """
        We apply the data screening apporach described in the AMAB paper
        which uses the median of the entries in the fixed window size.
    """
    i = 0
    # Initialize an empty list to store moving median
    moving_median = []

    while i < len(data) - window_size + 1:

        # Calculate the average of current window
        window_median = np.median(data[i:i+window_size],axis=0)

        # Store the median of current window in moving median list
        moving_median.append(window_median)
        # Shift window to right by one position
        i += 1

    return np.array(moving_median)


def duplicate_skeleton(arr):
    """
        For actions having only one human, we are replacing the zero matrix in the
        second skeleton with a copy of the node features for the first skeleton.
    """
    batch,timeLength,num_nodes,persons,in_dim = arr.shape
    count = 0
    for each in arr:
        in_count = 0
        for j in range(timeLength):
            diff = each[j,:,0,:] - each[j,:,1,:]
            condition = np.equal(diff, each[j,:,0,:]).sum() == (num_nodes*in_dim)
            if condition:
                in_count+= 1
        if in_count == timeLength :
            each[:,:,1,:] = each[:,:,0,:]
            count += 1

    print(f"{count} samples involved a single human.")



def load_data_from_file(train_file,test_file,adj_file=None,max_class_file=None,kinetics=False,dup_file=None):
    """
        loads the data from the provided file path as a binary.
    """
    if kinetics:
        train_file_data, train_file_label = train_file
        test_file_data, test_file_label = test_file

        with open(dup_file,'r') as f:
            duplicated = int(f.readline())

        # Data
        with open(train_file_data,'rb') as f:
            train_data = np.load(f,allow_pickle=True)
            train_data = np.transpose(train_data, (0, 2, 3, 4, 1))

        with open(test_file_data,'rb') as f:
            test_data = np.load(f,allow_pickle=True)
            test_data = np.transpose(test_data, (0, 2, 3, 4, 1))

        if not duplicated:
            print("\nDuplicating train data skeleton...")
            duplicate_skeleton(train_data)
            print("\nDuplicating test data skeleton...")
            duplicate_skeleton(test_data)

            # print("Saving duplicated data...")
            # with open(train_file_data,'wb') as f:
            #     np.save(f,train_data)
            # with open(test_file_data,'wb') as f:
            #     np.save(f,test_data)
            # with open(dup_file,'w') as f:
            #     f.write('1')
            #     print("Saving done!")

        # Reshape data
        # Train data
        b,t,N,p,d = train_data.shape
        train_data = train_data.reshape(b,t,-1,d)
        # Test data
        b,t,N,p,d = test_data.shape
        test_data = test_data.reshape(b,t,-1,d)

        # Labels
        with open(train_file_label, 'rb') as f:
            train_label = pickle.load(f)[1] #The second entry contains the labels

        with open(test_file_label, 'rb') as f:
            test_label = pickle.load(f)[1] #The second entry contains the labels

        return train_data, train_label, test_data, test_label

    # For the NTU RGB+D Dataset
    # with open(train_file,'rb') as f:
    #     train_data = np.load(f,allow_pickle=True)
    # with open(test_file,'rb') as f:
    #     test_data = np.load(f,allow_pickle=True)
    with open(adj_file,'rb') as f:
        A = np.load(f,allow_pickle=True)
    with open(max_class_file,'rb') as f:
        max_class = np.load(f)
    return None, None, A, max_class*1 #*1 makes sure it is scalar and not an array



def current_batch(train_graph,train_label,slice,iterations,batch_size):
    """
        This function returns the data--dtype(list) in batches.
    """
    if slice != iterations:
        a = slice*batch_size
        b = (slice+1)*batch_size
        return train_graph[a:b] , train_label[a:b]
    else:
        a = slice*batch_size
        return train_graph[a:] , train_label[a:]



def get_multi_view(arr,N=8):
    """This function outputs N rotations of the input (skeletal graph)
        about the x-z axis.
        arr: batch x time x nodes x feat_dim
        time= 300 (This can change across dataset)
        nodes= 25 (This can change across dataset)
        feat_dim=3 (This is fixed across dataset: x-y-z)

        output: [Sample1(views x t x n x f), Sample2(views x t x n x f),...,Sample_batchSize(views x t x n x f)]
    """
    if N == 1:
        return arr

    rotation_func = lambda n: np.array([[np.cos(2*np.pi*n/N), 0, np.sin(2*np.pi*n/N)],
                                        [0, 1, 0],
                                        [-np.sin(2*np.pi*n/N), 0, np.cos(2*np.pi*n/N)]
                                        ])

    rotation_matrix = np.array(list( map( rotation_func, range(1,N) ) )) # [N-1 x feat_dim x feat_dim]
    arr_reshape = arr.reshape(-1,arr.shape[-1]) # batch*time*nodes x feat_dim
    multiply = np.array(list(map( lambda i: np.matmul(rotation_matrix[i], arr_reshape.transpose()), range(N-1) )))
    # multiply : [N-1 x feat_dim x batch*time*nodes]

    multiply = multiply.transpose((0,2,1)) # [N-1 x batch*time*nodes x feat_dim]
    multiply = multiply.reshape(N-1,*arr.shape) # [N-1 x batch x time x nodes x feat_dim]
    result = np.concatenate( (np.expand_dims(arr, axis=0), multiply ) ) # [N x batch x time x nodes x feat_dim]
    result = result.transpose((1,0,2,3,4)) # [batch x N x time x nodes x feat_dim]
    _,_,*rest = result.shape
    return result.reshape(-1,*rest) # [batch*N x time x nodes x feat_dim]


def increase_label_with_views(label,N=8):
    """This function increases the number of labels to match the pattern of the multi-view output.
        output: [label1(views), label1(views), ... , label_batchSize(views)].flatten()
    """
    if N == 1:
        return label
    batch_size = len(label)
    new_label = list( map( lambda l : N*[l] , label) )

    return list(np.concatenate(new_label).flat) # type list of length N*len(label)



def train(train_graph,train_label,A,model,optimizer,criterion,device,numbers,
        topk=(None,),unsupervised=False,writer=None,count_flop=False):
    """
        This function has all the necessary modules to train our model.
    """
    if count_flop: #to count flops of model
        prof = FlopsProfiler(model)
        profile_step = 5
        print_profile= True

    if topk[0] is not None:
        res = []
        for i in topk:
            res.append(0) #Array to hold all the correct topk predictions
    batch_size,epoch = numbers
    # iterations = int(len(train_graph)/batch_size) #The number of iterations to exhaust our data when split into batches
    # pbar = tqdm(range(iterations + 1), unit='batch') # +1 is to ensure we reach the end of the list
    pbar = tqdm(train_graph)
    len_train_graph = len(train_graph.dataset) #length of train graph
    correct_pred = 0
    total_loss = 0
    model.train()

    for step,(feature, label, index) in enumerate(pbar):
    # for slice,pos in enumerate(pbar):
        with torch.no_grad():
            feature = feature.float().to(device)
            label = label.long().to(device)

        # train_graph_batch,train_label_batch = current_batch(train_graph,train_label,slice,iterations,batch_size)
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.

        # train_graph_batch = np.array(train_graph_batch)
        num_of_views = 1 #8
        # feature = torch.FloatTensor(get_multi_view(train_graph_batch,N=num_of_views)).to(device)
        if count_flop:
        # start profiling at training step "profile_step"
            if step == profile_step:
                prof.start_profile()
        output = model(feature) if A is None else model(feature,A,epoch)

        # train_label_batch = np.array(train_label_batch)
        # label = torch.LongTensor(increase_label_with_views(train_label_batch,N=num_of_views)).to(device)
        # Compute loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
        if unsupervised:
            out, *others = output
            loss = criterion(*(epoch, out, label, *others))

            output = out
        else:
            loss = criterion(output,label)
        if optimizer is not None:
            # Zero the gradients before running the backward pass.
            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. Internally, the parameters of each Module are stored
            # in Tensors with requires_grad=True, so this call will compute gradients for
            # all learnable parameters in the model.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if count_flop:
            # end profiling and print output
                if step == profile_step: # if using multi nodes, check global_rank == 0 as well
                    prof.stop_profile()
                    flops = prof.get_total_flops()
                    macs = prof.get_total_macs()
                    params = prof.get_total_params()
                    if print_profile:
                        print()
                        print(f"Total flops:{flops}")
                        # prof.print_model_profile(profile_step=profile_step)
                    prof.end_profile()
                    exit()
        total_loss += toNumpy(loss,device)
        # report
        pbar.set_description('epoch %d' % (epoch+1))
        # Accuracy

        pred = output.max(1, keepdim=True)[1]
        correct_pred += pred.eq(label.view_as(pred)).sum().detach().cpu().item()

        if topk[0] is not None:
            correct_preds_for_batch = topk_accuracy(output,label,topk=topk)

            for i,k in enumerate(topk):
                correct_k = correct_preds_for_batch[:k].reshape(-1).float().sum(0)
                res[i] = res[i] + correct_k

        #return average loss and accuracy of training data
    ave_loss = total_loss / float(len_train_graph)

    percent_acc = correct_pred/float(len_train_graph)

    if topk[0] is not None:
        for i,k in enumerate(topk):
            acc = res[i]/float(len_train_graph)
            print(f"top {k} accuracy is {acc:.6f}")
            if writer is not None:
                writer.add_scalar(f'Top {k}-Accuracy (train)', acc, epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

    if writer is not None:
        writer.add_scalar('Loss (train)', ave_loss, epoch)

    return ave_loss , percent_acc




def test(model, test_graphs, test_label, A, device, batch_size, criterion=None,
        topk=(None,), get_print=True, want_result=False, unsupervised=False,
        writer=None, return_accuracy=False):
    """
        This function helps to check the accuracy of the test data.
        ie. how well the model responds to unseen data.
    """
    if topk[0] is not None:
        res = []
        for i in topk:
            res.append(0) #Array to hold all the correct topk predictions

    if type(batch_size) is not int:
        batch_size, epoch = batch_size
    else:
        epoch = None

    model.eval()
    # iterations = int(len(test_graphs)/batch_size)
    total_loss = 0
    correct = 0
    if want_result:
        pred_result = []
        gt_labels = []

    # pbar = tqdm(range(iterations + 1), unit='batch') # +1 is to ensure we reach the end of the list
    pbar = tqdm(test_graphs)
    len_test_graph = len(test_graphs.dataset)

    for feature, label, index in pbar:
    # for slice,pos in enumerate(pbar):
        with torch.no_grad():
            feature = feature.float().to(device)
            label = label.long().to(device)

        # test_graph_batch,test_label_batch = current_batch(test_graphs,test_label,slice,iterations,batch_size)
        # test_graph_batch = np.array(test_graph_batch)
        # feature = torch.FloatTensor(test_graph_batch).to(device)

        # A = get_gaussian(feature,Adj) #Always compute the matrix

        output = model(feature) if A is None else model(feature,A)
        # test_label_batch = np.array(test_label_batch)
        # label = torch.LongTensor(test_label_batch).to(device)

        if unsupervised:
            out, *others = output
            loss = criterion(*(epoch, out, label, *others))

            output = out
        else:
            if criterion is not None:
                loss = criterion(output,label)

        # Accuracy
        if (topk[0] is None) or want_result:
            pred = output.max(1, keepdim=True)[1]
            if want_result:
                pred_result.append(toNumpy(pred,device))
                gt_labels.append(toNumpy(label,device))
            correct += pred.eq(label.view_as(pred)).sum().item()

        if topk[0] is not None:
            correct_preds_for_batch = topk_accuracy(output,label,topk=topk)

            for i,k in enumerate(topk):
                correct_k = correct_preds_for_batch[:k].reshape(-1).float().sum(0)
                res[i] = res[i] + correct_k
        # Loss
        # loss = criterion(output,label)
        if criterion is not None:
            total_loss += toNumpy(loss,device)
        #report
        pbar.set_description('Testing')

    if criterion is not None:
        loss_test = total_loss / float(len_test_graph)
    else:
        loss_test = total_loss

    if topk[0] is None:
        acc_test = correct / float(len_test_graph)

    if get_print:
        if topk[0] is None:
            # print(f"acc_test: {acc_test:.4f}, loss_test: {loss_test:.4f}")
            print("acc_test: %f, loss_test: %f" %(acc_test, loss_test))
        else:
            print("loss_test: %f" %(loss_test))
            for i,k in enumerate(topk):
                acc = res[i]/float(len_test_graph)
                print(f"top {k} accuracy is {acc:.6f}*********")
                if (writer is not None) and (epoch is not None):
                    writer.add_scalar(f'Top {k}-Accuracy (test)', acc, epoch)

            if (writer is not None) and (epoch is not None):
                writer.add_scalar('Loss (test)', loss_test, epoch)

    if return_accuracy:
        return loss_test, (res[0]/float(len_test_graph)).cpu().item()

    if want_result:
        return pred_result, gt_labels


def topk_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    return correct


def validation_fn(model, test_graphs, test_label, A, device, criterion, batch_size, unsupervised=False):
    """
        This function returns the loss of the validation/test data.
    """
    model.eval()
    iterations = int(len(test_graphs)/batch_size)
    total_loss = 0
    for slice in range(iterations + 1): # +1 is to ensure we reach the end of the list
        test_graph_batch,test_label_batch = current_batch(test_graphs,test_label,slice,iterations,batch_size)
        test_graph_batch = np.array(test_graph_batch)
        feature = torch.FloatTensor(test_graph_batch).to(device)

        # A = get_gaussian(feature,Adj) #Always compute the matrix

        output = model(feature) if A is None else model(feature,A)
        test_label_batch = np.array(test_label_batch)
        label = torch.LongTensor(test_label_batch).to(device)
        loss = criterion(output,label)
        total_loss += toNumpy(loss,device)
    return total_loss / float(len(test_graphs))
