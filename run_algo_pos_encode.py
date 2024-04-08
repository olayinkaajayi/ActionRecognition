import torch
import torch.optim as optim
import sys
import os
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

sys.path.append("DHCS_implement/models")
sys.path.append("DHCS_implement/models/Positional_encoding")
# from pos_encode import Position_encode
from pos_encode_xx import Position_encode
from pos_encode_loss import Position_encode_loss
from training_utils_py2 import use_cuda, toNumpy

from Improved_model.arg_n_utils import arg_parse, get_labels



def main():
    print()
    args = arg_parse()

    if args.use_kinetics:
        folder = '/dcs/large/u2034358/'
        adj_file = 'WikiCSDataset_adj.npy'

        with open(folder+adj_file,'rb') as f:
            A = np.load(f,allow_pickle=True)

        print("Working on the WikiCS Graph dataset")

    else:
        _, _, A, _ = get_labels(args.cross_)
    # A = np.array([[0,1,1,1,0], #sample graph
    #               [1,0,1,0,0],
    #               [1,1,0,0,0],
    #               [1,0,0,0,1],
    #               [0,0,0,1,0]])

    # A = np.array([[0,1,1,1,1], #sample graph
    #               [1,0,0,0,0],
    #               [1,0,0,0,0],
    #               [1,0,0,0,0],
    #               [1,0,0,0,0]])


    # A = np.array([[0,1,0,0,0], #sample graph
    #               [1,0,1,0,0],
    #               [0,1,0,1,1],
    #               [0,0,1,0,0],
    #               [0,0,1,0,0]])

    # A = np.array([[0,1,0,0], #sample graph
    #               [1,0,1,0],
    #               [0,1,0,1],
    #               [0,0,1,0]])

    #Initialize seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ########Load data################
    begin_path = os.getcwd()
    many_gpu = args.many_gpu #Decide if we use multiple GPUs or not
    device,USE_CUDA = use_cuda(args.use_cpu,many=many_gpu)

    #load model
    d = 8
    N = A.shape[0]
    old_similar = A.shape[0]
    print(f"Matrix is size {N}x{N}")
    # d = 7
    # old_similar = 25
    old_loss = np.inf
    model = Position_encode(A=A, d=d)

    if args.use_saved_model:
        # To load model
        model.load_state_dict(torch.load(begin_path+f'/DHCS_implement/Saved_models/d={d}'+args.checkpoint,map_location=device))
        print("USING SAVED MODEL!")


    if USE_CUDA: #To set it up for parallel usage of both GPUs (speeds up training)
        torch.cuda.manual_seed_all(args.seed)
        model = torch.nn.DataParallel(model) if many_gpu else model #use all free GPUs if needed
    model.to(device)

    criterion = Position_encode_loss(k1=1.0, k2=1.0)

    params = list(model.parameters())
    Num_Param = sum(p.numel() for p in params if p.requires_grad)
    print("Number of Trainable Parameters is about %d" % (Num_Param))

    optimizer = optim.Adam(params, lr= args.lr)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                              factor=0.1,
#                                                              patience=10,
#                                                              verbose=True, min_lr=0.0001)
# #########################################################
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10], gamma=0.1)

    print("Pre-training Result:")
    numbering, similar = check_result(model, device)#, final=True)
    print(f"--{similar} nodes are similar\n")

    collector = []

    filename = save_as_txt(numbering[0], device, args.seed, encounter=0, filename=args.checkpoint[:-3], name_return=True, gather=collector) # To get 1-25
    save_as_txt(numbering[1], device, args.seed, filename=args.checkpoint[:-3], gather=collector) # To get pretrained order
    print(f"Numbering:\n{toNumpy(numbering,device)}")

    # for epoch in range(args.epochs):
    with tqdm(range(args.epochs)) as t:
        for epoch in t:

            t.set_description('Epoch %d' % (epoch+1))
            model.train()
            total_loss = 0
            out1, out2, out3 = model(torch.arange(N))
            loss = criterion(out1.sum(), out2.sum(), out3.sum())

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss = toNumpy(loss,device)

            # scheduler.step(loss.item())

            with torch.no_grad():
                numbering, similar = check_result(model, device)
                print(f"Epoch {epoch+1}: loss= {total_loss}")
                print(f"--{similar} nodes are similar")
                if determine_save(total_loss, old_loss, similar, old_similar, epoch, min_runs=d):
                    old_similar = similar
                    old_loss = total_loss
                    save_as_txt(numbering[1], device, args.seed, filename=args.checkpoint[:-3], gather=collector)
                    print(f"Numbering:\n{toNumpy(numbering,device)}")

                    if USE_CUDA and many_gpu:
                        torch.save(model.module.state_dict(), begin_path+f'/DHCS_implement/Saved_models/d={d}'+args.checkpoint)
                    else:
                        torch.save(model.state_dict(), begin_path+f'/DHCS_implement/Saved_models/d={d}'+args.checkpoint)

    # Load best result
    del model
    print("\nGetting result from SAVED MODEL:")
    model = Position_encode(A=A, d=d)
    model.load_state_dict(torch.load(begin_path+f'/DHCS_implement/Saved_models/d={d}'+args.checkpoint,map_location=device))
    if USE_CUDA: #To set it up for parallel usage of both GPUs (speeds up training)
        torch.cuda.manual_seed_all(args.seed)
        # model = torch.nn.DataParallel(model) if many_gpu else model #use all free GPUs if needed
    model.to(device)

    output = model(torch.arange(N).to(device))
    loss = criterion(*output)
    total_loss = toNumpy(loss,device)

    gen_plot_change(collector, filename) #To know the nodes that change during training

    numbering, similar = check_result(model, device, final=True)

    print("\nBest Result is:\n")
    print(f"loss= {total_loss}")
    print(f"--{similar} nodes are similar")
    print(f"Numbering:\n{toNumpy(numbering,device)}")

    with open(filename,'a') as f:
        f.write(f"\nFinal loss={total_loss:.2f}\n")
        f.write(f"\nFinal similar={similar}")
        f.write("\n")

    # END algo


def determine_save(loss, old_loss, new_similar, old_similar, epoch, min_runs):
    """Decide if the model paramter should be saved"""
    return (loss <= old_loss) and (new_similar <= old_similar) and (epoch > min_runs)


def check_result(model, device, final=False):
    """Returns the numbering of the skeletal graph"""
    model.eval()

    Z, pred_degree, degree = model(test=True,deg=final)
    N,d = Z.shape

    # This is because we use multiple GPU
    if not final:
        N = N//torch.cuda.device_count()
    Z = Z[:N]
    err = 0.0001
    Z = Z + err
    Z = torch.round(Z)
    two_vec = torch.zeros(d).to(device)
    for i in range(d):
        two_vec[i] = pow(2,d-1-i)
    numbers = (Z * two_vec).sum(dim=1).unsqueeze(0) #shape: N x 1

    num_list = toNumpy(numbers.squeeze(0),device).tolist()
    num_set = set(num_list)
    diff = (len(num_list) - len(num_set))*2

    nodes = torch.arange(1,N+1).unsqueeze(0).to(device)
    numbers= torch.cat([nodes, numbers]).long()

    if final:
        pred_degree, degree = pred_degree[:N], degree[:N]
        get_plot(Z, device, d)
        pred_deg = toNumpy(torch.cat([degree.unsqueeze(0).to(device), pred_degree.unsqueeze(0)]), device)
        print(f"Degree:\n{pred_deg}")

    return numbers, diff


def get_plot(Z, device, d):
    """Returns a gray-scale of the position encoding"""
    plt.figure(figsize=(40,24))
    plt.imshow(toNumpy(Z.t(),device), cmap=cm.Greys_r)
    plt.xticks(range(Z.shape[0]), list(range(1,Z.shape[0]+1)), fontsize=54)
    plt.tick_params(left = False , labelleft = False)
    plt.title(f'Binary Position Encoding, d={d}', fontdict={'fontsize': 80})
    plt.savefig(os.getcwd()+f'/Pos_Enc_image/pos_encode_dim={d}.png',
                bbox_inches='tight')
    print('plot saved...')


def gen_plot_change(collector, filename):
    """Returns a black and white of where a nodes numbering changed."""
    collector = check_entries(np.array(collector)[-1::-1])
    plt.imshow(collector, cmap=cm.Greys_r)
    plt.xticks(range(collector.shape[1]), list(range(1,collector.shape[1]+1)))
    plt.tick_params(left = False , labelleft = False)
    plt.savefig(f'{filename[:-4]}.png', bbox_inches='tight')
    print('Change-plot saved...')


def save_as_txt(numbering, device, seed, encounter=1, filename='NAPE_numbers', name_return=False, gather=[]):

    number = toNumpy(numbering,device)
    if encounter != 0:
        gather.append(number)

    if encounter == 0:
        task = 'w+'
    else:
        task = 'a'

    filename = filename + '_' + 'seed_' + str(seed) + '_' + 'device_' + str(device) + '.txt'

    with open(filename,task) as f:
        for entry in number:
            f.write(f"{int(entry):>3}"+'\t')
        f.write("\n")

        if encounter == 0:
            f.write(f"{'':->105}")
            f.write("\n")

    if name_return:
        return filename


def check_entries(matrix):
    # Convert matrix to numpy array
    matrix = np.array(matrix)

    # Get the number of rows and columns in the matrix
    T, N = matrix.shape

    # Initialize an empty array to store the results
    results = np.empty((T-1, N), dtype=int)

    # Iterate over each row, except the last one
    for row in range(T-1):
        # Compare the entries between consecutive rows for each column
        results[row] = np.where(matrix[row, :] != matrix[row+1, :], 1, 0)

    return results


if __name__ == '__main__':
    main()
    print("Done!!!")
