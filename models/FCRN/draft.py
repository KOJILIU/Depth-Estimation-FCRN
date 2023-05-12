import torch
import torch.utils.data
import torchvision
from loader import *
import os
from fcrn import FCRN
from torch.autograd import Variable
from weights import load_weights
import matplotlib
# without picture show
matplotlib.use('Agg')
import matplotlib.pyplot as plot

dtype = torch.cuda.FloatTensor
weights_file = "NYU_ResNet-UpProj.npy"


# return train_lists, val_lists, test_lists
def load_split():
    current_directoty = os.getcwd()
    train_lists_path = current_directoty + '/trainIdxs.txt'
    test_lists_path = current_directoty + '/testIdxs.txt'

    train_f = open(train_lists_path)
    test_f = open(test_lists_path)

    train_lists = []
    test_lists = []
    # 一行一行的读数据
    train_lists_line = train_f.readline()
    while train_lists_line:
        train_lists.append(int(train_lists_line) - 1)
        train_lists_line = train_f.readline()
    train_f.close()

    test_lists_line = test_f.readline()
    while test_lists_line:
        test_lists.append(int(test_lists_line) - 1)
        test_lists_line = test_f.readline()
    test_f.close()
    # 0.2 的测试集
    val_start_idx = int(len(train_lists) * 0.8)

    val_lists = train_lists[val_start_idx:-1]
    train_lists = train_lists[0:val_start_idx]

    return train_lists, val_lists, test_lists
# 返回测试数据（盲盒），训练数据，测试集数据（都是列表）

def main():
    batch_size = 32
    data_path = 'nyu_depth_v2_labeled.mat'

    # 1.Load data
    train_lists, val_lists, test_lists = load_split()
    print("Loading data......")
    # 就是个加载数据的loader（dataset batch shuffle：true在每个epoch数据重组，drop_last删去不能除尽的组）
    train_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, train_lists),
                                               batch_size=batch_size, shuffle=True, drop_last=True)
    # val_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, val_lists),
    #                                          batch_size=batch_size, shuffle=True, drop_last=True)
    # test_loader = torch.utils.data.DataLoader(NyuDepthLoader(data_path, test_lists),
    #                                           batch_size=batch_size, shuffle=True, drop_last=True)
    print(train_loader)
    num = 0
    for input, depth in train_loader:
        input_var = Variable(input.type(dtype))
        depth_var = Variable(depth.type(dtype))
        print(len(input_var))

        #input_rgb_image = input_var[0].data.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        #input_gt_depth_image = depth_var[0][0].data.cpu().numpy().astype(np.float32)

        #input_gt_depth_image /= np.max(input_gt_depth_image)

        #plot.imsave('./draft/input_rgb_epoch_{}.png'.format(num), input_rgb_image)
        #plot.imsave('./draft/gt_depth_epoch_{}.png'.format(num), input_gt_depth_image, cmap="viridis")
        num += 1

if __name__ == '__main__':
    main()