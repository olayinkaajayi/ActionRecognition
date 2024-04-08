import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from tqdm import tqdm
import torch_geometric
import os
import os.path as osp
from torch_geometric.data import InMemoryDataset, Data

#%% Dataset to manage vector to vector data

class NTURGB_view(InMemoryDataset):

    def __init__(self, root, idx, transform=None, pre_transform=None):
        super(NTURGB_view, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        # List of the raw files
        return ['cross_view_train.npy','cross_view_test.npy','adj_matrix.npy','max_class.npy']

    @property
    def processed_file_names(self):
        return ['cross_view_data_train.pt','cross_view_data_test.pt'] #Add these names for cross_view test and train.

    def process(self):
        # Read the files' content as Pandas DataFrame. Nodes and graphs ids
        # are based on the file row-index, we adjust the DataFrames indices
        # by starting from 1 instead of 0.
        cross_view_train_file = self.root+'/cross_view_train.npy'
        cross_view_test_file = self.root+'/cross_view_test.npy'
        adj_file = self.root+'/adj_matrix.npy'
        max_class_file = self.root+'/max_class.npy'
        train_data, test_data, A, num_class = load_data_from_file(cross_view_train_file, cross_view_test_file, adj_file, max_class_file)

        # Get labels
        labels_train = [int(ele['class']) for ele in train_data]
        labels_test = [int(ele['class']) for ele in test_data]

        # Get maximum time length
        max_time_train = max([ele['skel_body0'].shape[0] for ele in train_data])
        max_time_test = max([ele['skel_body0'].shape[0] for ele in test_data])
        max_time = max(max_time_test,max_time_train)

        #Pads the data to be of equal timeLength
        train_data = pad_data(train_data,max_time)
        test_data = pad_data(test_data,max_time)
        # In the loop we extract the nodes' embeddings, edges connectivity for
        # and label for a graph, process the information and put it in a Data
        # object, then we add the object to a list
        for index,data in enumerate([train_data,test_data]):
            data_list = []
            data_len,time_len,num_nodes,in_dim = np.shape(data)
            ids_list = range(data_len)
            for g_idx in tqdm(ids_list):
                # Node features
                attributes = np.reshape(data[g_idx],(-1,in_dim))

                # Edges info
                if g_idx == 0:
                    edge = edge_index(A)
                    comb_graph = [i*num_nodes + edge for i in range(time_len)] #combining the graph across time domain
                    edge_idx = torch.cat(comb_graph,dim=-1)
                else:
                     edge_idx = data_list[0].edge_index

                # Graph label
                label = labels_train[g_idx] if index == 0 else labels_test[g_idx]

                # Convert the numpy array into tensors
                x = torch.tensor(attributes, dtype=torch.float)

                y = torch.tensor(label, dtype=torch.long)

                graph = Data(x=x, edge_index=edge_idx,  y=y)

                data_list.append(graph)

            # Apply the functions specified in pre_filter and pre_transform
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            # Store the processed data
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[index])

class NTURGB_sub(InMemoryDataset):

    def __init__(self, root, idx, transform=None, pre_transform=None):
        super(NTURGB_sub, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[idx])

    @property
    def raw_file_names(self):
        # List of the raw files
        return ['cross_sub_train.npy','cross_sub_test.npy','adj_matrix.npy','max_class.npy']

    @property
    def processed_file_names(self):
        return ['cross_sub_data_train.pt','cross_sub_data_test.pt'] #Add these names for cross_view test and train.

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'cross_sub_data_train.pt' if idx==0 else 'cross_sub_data_test.pt'))
        return data

    def process(self):
        # Read the files' content as Pandas DataFrame. Nodes and graphs ids
        # are based on the file row-index, we adjust the DataFrames indices
        # by starting from 1 instead of 0.
        cross_view_train_file = self.root+'/cross_sub_train.npy'
        cross_view_test_file = self.root+'/cross_sub_test.npy'
        adj_file = self.root+'/adj_matrix.npy'
        max_class_file = self.root+'/max_class.npy'
        train_data, test_data, A, num_class = load_data_from_file(cross_view_train_file, cross_view_test_file, adj_file, max_class_file)

        # Get labels
        labels_train = [int(ele['class']) for ele in train_data]
        labels_test = [int(ele['class']) for ele in test_data]

        # Get maximum time length
        max_time_train = max([ele['skel_body0'].shape[0] for ele in train_data])
        max_time_test = max([ele['skel_body0'].shape[0] for ele in test_data])
        max_time = max(max_time_test,max_time_train)

        #Pads the data to be of equal timeLength
        train_data = pad_data(train_data,max_time)
        test_data = pad_data(test_data,max_time)
        # In the loop we extract the nodes' embeddings, edges connectivity for
        # and label for a graph, process the information and put it in a Data
        # object, then we add the object to a list
        for index,data in enumerate([train_data,test_data]):
            data_list = []
            data_len,time_len,num_nodes,in_dim = np.shape(data)
            ids_list = range(data_len)
            for g_idx in tqdm(ids_list):
                # Node features
                attributes = np.reshape(data[g_idx],(-1,in_dim))

                # Edges info
                if g_idx == 0:
                    edge = edge_index(A)
                    comb_graph = [i*num_nodes + edge for i in range(time_len)] #combining the graph across time domain
                    edge_idx = torch.cat(comb_graph,dim=-1)
                else:
                     edge_idx = data_list[0].edge_index

                # Graph label
                label = labels_train[g_idx] if index == 0 else labels_test[g_idx]

                # Convert the numpy array into tensors
                x = torch.tensor(attributes, dtype=torch.float)

                y = torch.tensor(label, dtype=torch.long)

                graph = Data(x=x, edge_index=edge_idx,  y=y)

                data_list.append(graph)

            # Apply the functions specified in pre_filter and pre_transform
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            # Store the processed data
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[index])

def work_data(begin_path,work_with_data):
    file_path = begin_path+'/DHCS_implement/work_with.npy'
    with open(file_path,'wb') as q:
        np.save(q,work_with_data)

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
                print(f"caught an error %d "%(cnt))
                continue
    return total


def edge_index(A):
    """
        A: Adjacency matrix --dim(num_nodes,num_nodes)
    """
    top,bottom = np.where(A==1)
    q = np.concatenate((top.reshape(1,-1),bottom.reshape(1,-1)))
    return torch.from_numpy(q).type(torch.LongTensor)



def use_cuda(use_cpu,many=False):
    """
        Imports GPU if available, else imports CPU.
    """
    if use_cpu: #overrides GPU and use CPU
        device = torch.device("cpu")
        USE_CUDA = False
        print("CPU used")
        return device,USE_CUDA

    USE_CUDA = torch.cuda.is_available()
    if USE_CUDA:
        device = torch.device("cuda" if many else "cuda:0")
        print("GPU is available")
    else:
        device = torch.device("cpu")
        print("GPU not available, CPU used")
    return device,USE_CUDA

def toTensor(v,device,dtype = torch.float,requires_grad = False):#This function converts the entry 'v' to a tensor data type
    return Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad).to(device)

def toNumpy(v,device): #This function converts the entry 'v' back to a numpy (float) data type
    if device.type == 'cuda':
        return v.detach().cpu().numpy()
    return v.detach().numpy()

def pad_data(dat,max_time=300):
    """
        dat is a dictionary.
        This function pads the time dimension of dat with a 25 x 3 zero matrix.
    """

    dat_holder_across_time = []
    for each in dat:
        if each['skel_body0'].shape[0] < max_time:
            dat_time,row,col = each['skel_body0'].shape
            new_array = np.zeros((max_time,row,col))
            new_array[:dat_time] = each['skel_body0']
            # Cannot have this guy. Need matrix to be same dimension for all
            # new_array[-1,-1,-1] = dat_time #This would hold the true length of video frames
            each['skel_body0'] = new_array
        dat_holder_across_time.append(each['skel_body0'])
    return dat_holder_across_time


def load_data_from_file(train_file,test_file,adj_file,max_class_file):
    """
        loads the data from the provided file path as a binary.
    """
    with open(train_file,'rb') as f:
        train_data = np.load(f,allow_pickle=True)
    with open(test_file,'rb') as f:
        test_data = np.load(f,allow_pickle=True)
    with open(adj_file,'rb') as f:
        A = np.load(f,allow_pickle=True)
    with open(max_class_file,'rb') as f:
        max_class = np.load(f)
    return train_data, test_data, A, max_class*1 #*1 makes sure it is scalar and not an array


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


def train(train_graph,train_label,A,model,optimizer,criterion,device,numbers):
    """
        This function has all the necessary modules to train our model.
    """
    batch_size,epoch = numbers
    iterations = int(len(train_graph)/batch_size) #The number of iterations to exhaust our data when split into batches
    pbar = tqdm(range(iterations + 1), unit='batch') # +1 is to ensure we reach the end of the list
    correct_pred = 0
    total_loss = 0
    model.train()

    for feature,label in train_loader:
        # Forward pass: compute predicted y by passing x to the model. Module objects
        # override the __call__ operator so you can call them like functions. When
        # doing so you pass a Tensor of input data to the Module and it produces
        # a Tensor of output data.
        if device.type == 'cuda':
            feature = torch.FloatTensor(train_graph_batch).cuda()
        else:
            feature = torch.FloatTensor(train_graph_batch).to(device)
        output = model(feature,A)
        if device.type == 'cuda':
            label = torch.LongTensor(train_label_batch).cuda()
        else:
            label = torch.LongTensor(train_label_batch).to(device)
        # Compute loss. We pass Tensors containing the predicted and true
        # values of y, and the loss function returns a Tensor containing the
        # loss.
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
        total_loss += toNumpy(loss,device)
        # report
        pbar.set_description('epoch: %d' % (epoch+1))
        pred = output.max(1, keepdim=True)[1]
        correct_pred += pred.eq(label.view_as(pred)).sum().detach().cpu().item()
        # correct_pred += (torch.argmax(output) == label)

        #return average loss and accuracy of training data
    return total_loss / float(len(train_graph)) , correct_pred/float(len(train_graph))


def validation_fn(model, test_graphs, test_label, A, device, criterion, batch_size):
    """
        This function returns the loss of the validation/test data.
    """
    model.eval()
    iterations = int(len(test_graphs)/batch_size)
    total_loss = 0
    for slice in range(iterations + 1): # +1 is to ensure we reach the end of the list
        test_graph_batch,test_label_batch = current_batch(test_graphs,test_label,slice,iterations,batch_size)
        if device.type == 'cuda':
            feature = torch.FloatTensor(test_graph_batch).cuda()
        else:
            feature = torch.FloatTensor(test_graph_batch).to(device)
        output = model(feature,A)
        if device.type == 'cuda':
            label = torch.LongTensor(test_label_batch).cuda()
        else:
            label = torch.LongTensor(test_label_batch).to(device)
        loss = criterion(output,label)
        total_loss += toNumpy(loss,device)
    return total_loss / float(len(test_graphs))


def test(model, test_graphs, test_label, A, device, batch_size, want_result=False):
    """
        This function helps to check the accuracy of the test data.
        ie. how well the model responds to unseen data.
    """
    model.eval()
    batch_size = 50 #to reduce load
    iterations = int(len(test_graphs)/batch_size)
    correct = 0
    for slice in range(iterations + 1): # +1 is to ensure we reach the end of the list
        test_graph_batch,test_label_batch = current_batch(test_graphs,test_label,slice,iterations,batch_size)
        if device.type == 'cuda':
            feature = torch.FloatTensor(test_graph_batch).cuda()
        else:
            feature = torch.FloatTensor(test_graph_batch).to(device)
        output = model(feature,A)
        if device.type == 'cuda':
            label = torch.LongTensor(test_label_batch).cuda()
        else:
            label = torch.LongTensor(test_label_batch).to(device)
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(label.view_as(pred)).sum().item()

    acc_test = correct / float(len(test_graphs))
    print("accuracy test: %f" % (acc_test))

    if want_result: #If we want result
        return acc_test, pred


def train_pyg(train_loader,model,optimizer,criterion,device,numbers):
    """
        This function has all the necessary modules to train our model.
    """
    batch_size,epoch = numbers
    pbar = tqdm(train_loader, unit='batch') # +1 is to ensure we reach the end of the list
    correct_pred = 0
    total_loss = 0
    model.train()
    iters = 0
    # for slice,pos in enumerate(pbar):
    for data_list in pbar:
        output = model(data_list)#.to(device))
        y = torch.cat([data.y for data in data_list]).to(device)
        loss = criterion(output,y)
        # loss = criterion(output,data.y)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += toNumpy(loss,device)
        # report
        pbar.set_description('epoch %d' % (epoch+1))
        pred = output.max(1, keepdim=True)[1]
        correct_pred += pred.eq(y.view_as(pred)).sum().item()
        # correct_pred += pred.eq(data.y.view_as(pred)).sum().detach().cpu().item()
        #return average loss and accuracy of training data
    return total_loss / float(len(train_loader.dataset)) , correct_pred/float(len(train_loader.dataset))


def test_pyg(model, test_loader, device, want_result=False):
    """
        This function helps to check the accuracy of the test data.
        ie. how well the model responds to unseen data.
    """
    model.eval()
    pbar = tqdm(test_loader, unit='batch')
    correct = 0
    for data_list in pbar:
        output = model(data_list)#.to(device))
        pred = output.max(1, keepdim=True)[1]
        y = torch.cat([data.y for data in data_list]).to(device)
        correct += pred.eq(y.view_as(pred)).sum().item()
        # correct += pred.eq(data.y.view_as(pred)).sum().item()
        pbar.set_description('Testing')

    acc_test = correct / float(len(test_loader.dataset))
    print("accuracy test: %f" % (acc_test))

    if want_result: #If we want result
        return acc_test, pred
