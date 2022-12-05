import os
from unittest import defaultTestLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import shutil
import argparse
import numpy as np
import torch
import scipy.io as io

from core.models.model_factory import Model
from core.utils import preprocess
import core.trainer as trainer
from data_provider.CIKM.data_iterator import *
import math
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch video prediction model - EF_convlstm')

# test/training/use
parser.add_argument('--is_training', type=int, default=2)
parser.add_argument('--device', type=str, default='gpu:0')

# data
parser.add_argument('--is_parallel', type=bool, default=False)
parser.add_argument('--dataset_name', type=str, default='radar')
parser.add_argument('--save_dir', type=str, default='checkpoints/bitcq_ef_convlstm')
parser.add_argument('--gen_frm_dir', type=str, default='test/bitcq_ef_convlstm/')
parser.add_argument('--input_length', type=int, default=10)
parser.add_argument('--total_length', type=int, default=30)
parser.add_argument('--img_width', type=int, default=384)
parser.add_argument('--img_channel', type=int, default=1)

# model
parser.add_argument('--model_name', type=str, default='ef_convlstm')
parser.add_argument('--model_type', type=str, default='ef')
parser.add_argument('--pretrained_model', type=str, default='')
parser.add_argument('--patch_size', type=int, default=1)
parser.add_argument('--num_hidden', type=str, default='128, 128, 128')
parser.add_argument('--kernel', type=int, default=3)

# scheduled sampling
parser.add_argument('--scheduled_sampling', type=int, default=1)
parser.add_argument('--sampling_stop_iter', type=int, default=50000)
parser.add_argument('--sampling_start_value', type=float, default=1.0)
parser.add_argument('--sampling_changing_rate', type=float, default=0.00002)

# optimization
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--reverse_input', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_iterations', type=int, default=30000)
parser.add_argument('--display_interval', type=int, default=10)
parser.add_argument('--test_interval', type=int, default=1000)
parser.add_argument('--snapshot_interval', type=int, default=5000)
parser.add_argument('--num_save_samples', type=int, default=10)
parser.add_argument('--n_gpu', type=int, default=1)

# use
parser.add_argument('--temp_frm_dir', type=str, default='temp/bitcq_ef_convlstm/')
parser.add_argument('--origin_img_dir', type=str, default="")
parser.add_argument('--rain_threshold', type=float, default=0.1)


args = parser.parse_args()
batch_size = args.batch_size

def padding_Jiangsu_data(frame_data):
    shape = frame_data.shape
    batch_size = shape[0]
    seq_length = shape[1]
    padding_frame_dat = np.zeros((batch_size,seq_length,args.img_width,args.img_width,args.img_channel))
    padding_frame_dat[:, :, 40:-40, :, :] = frame_data
    return padding_frame_dat

def unpadding_Jiangsu_data(padding_frame_dat):
    return padding_frame_dat[:,:,40:-40,:,:]

def schedule_sampling(eta, itr):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if not args.scheduled_sampling:
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                           (args.batch_size,
                            args.total_length - args.input_length - 1,
                            args.img_width // args.patch_size,
                            args.img_width // args.patch_size,
                            args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag


def wrapper_use(model):
    model.network.eval()
    temp_save_root = args.temp_frm_dir
    clean_fold(temp_save_root)
    
    output_length = args.total_length - args.input_length
    real_input_flag = np.zeros(
        (1,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    
    dat = sample(1, data_type='use', root_path=args.origin_img_dir)
    
    dat = nor(dat)
    tars = dat[:, -output_length:]
    ims = preprocess.reshape_patch(dat, args.patch_size)
    img_gen,_ = model.use(ims, real_input_flag)
    img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
    
    img_out = de_nor(img_gen)
    
    os.mkdir(temp_save_root + 'img/')
    for t in range(0, 20, 1):
        imsave(temp_save_root + 'img/' + 'img_' + str(t + 11) + '.png', img_out[0, t, 40:-40, :, 0])
    
    os.mkdir(temp_save_root + 'mat')
    os.mkdir(temp_save_root + 'mat/TN')
    os.mkdir(temp_save_root + 'mat/FP')
    os.mkdir(temp_save_root + 'mat/FN')
    os.mkdir(temp_save_root + 'mat/TP')
    
    mat_out = img_gen * 70
    tar_pre = tars * 70
    TN = (tar_pre >= args.rain_threshold) & (mat_out >= args.rain_threshold)
    FP = (tar_pre < args.rain_threshold) & (mat_out >= args.rain_threshold)
    FN = (tar_pre >= args.rain_threshold) & (mat_out < args.rain_threshold)
    TP = (tar_pre < args.rain_threshold) & (mat_out < args.rain_threshold)
    for t in range(0, 20, 1):
        io.savemat(temp_save_root + 'mat/TN/' + 'img_' + str(t + 11) + '.mat', {'name':TN[0, t, 40:-40, :, 0]})
        io.savemat(temp_save_root + 'mat/FP/' + 'img_' + str(t + 11) + '.mat', {'name':FP[0, t, 40:-40, :, 0]})
        io.savemat(temp_save_root + 'mat/FN/' + 'img_' + str(t + 11) + '.mat', {'name':FN[0, t, 40:-40, :, 0]})
        io.savemat(temp_save_root + 'mat/TP/' + 'img_' + str(t + 11) + '.mat', {'name':TP[0, t, 40:-40, :, 0]})


def wrapper_test(model):
    model.network.eval()
    test_save_root = args.gen_frm_dir
    clean_fold(test_save_root)
    loss = 0
    count = 0
    index = 1
    flag = True
    img_mse, ssim = [], []

    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    real_input_flag = np.zeros(
        (1,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    output_length = args.total_length - args.input_length
    while flag:
        dat, (index, b_cup) = sample(1, data_type='test', index=index)

        dat = nor(dat)
        tars = dat[:, -output_length:]
        ims = preprocess.reshape_patch(dat, args.patch_size)
        img_gen,_ = model.test(ims, real_input_flag)
        img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
        img_out = img_gen

        mse = np.mean(np.square(tars - img_out))

        img_out = de_nor(img_out)
        loss = loss + mse
        count = count + 1

        bat_ind = 0
        for ind in range(index - 1, index, 1):
            save_fold = test_save_root + 'sample_' + str(ind) + '/'
            clean_fold(save_fold)
            for t in range(0, 20, 1):
                imsave(save_fold + 'img_' + str(t) + '.png', img_out[bat_ind, t, :, :, 0])
            bat_ind = bat_ind + 1
            
        if b_cup != -1:
            pass
        else:
            flag = False

    return loss / count


def wrapper_valid(model):
    model.network.eval()
    loss = 0
    count = 0
    index = 1
    flag = True
    img_mse, ssim = [], []

    for i in range(args.total_length - args.input_length):
        img_mse.append(0)
        ssim.append(0)

    real_input_flag = np.zeros(
        (args.batch_size,
         args.total_length - args.input_length - 1,
         args.img_width // args.patch_size,
         args.img_width // args.patch_size,
         args.patch_size ** 2 * args.img_channel))
    output_length = args.total_length - args.input_length
    while flag:

        dat, (index, b_cup) = sample(batch_size, data_type='validation', index=index)
        dat = nor(dat)
        tars = dat[:, -output_length:]
        ims = preprocess.reshape_patch(dat, args.patch_size)
        img_gen,_ = model.test(ims, real_input_flag)
        img_gen = preprocess.reshape_patch_back(img_gen, args.patch_size)
        img_out = img_gen[:, -output_length:]


        mse = np.mean(np.square(tars-img_out))
        loss = loss+mse
        count = count+1
        if b_cup != -1:
            pass
        else:
            flag = False

    return loss/count




def wrapper_train(model):
    model.network.train()
    if args.pretrained_model:
        model.load()


    eta = args.sampling_start_value
    best_mse = math.inf
    tolerate = 0
    limit = 3
    best_iter = None
    for itr in range(1, args.max_iterations + 1):

        ims = sample(
            batch_size=batch_size
        )
        ims = preprocess.reshape_patch(ims, args.patch_size)
        ims = nor(ims)
        
        eta, real_input_flag = schedule_sampling(eta, itr)

        cost = trainer.train(model, ims, real_input_flag, args, itr)

        if itr % args.display_interval == 0:
            print('itr: ' + str(itr))
            print('training loss: ' + str(cost))



        if itr % args.test_interval == 0:
            print('validation one ')
            valid_mse = wrapper_valid(model)
            print('validation mse is:',str(valid_mse))

            if valid_mse<best_mse:
                best_mse = valid_mse
                best_iter = itr
                tolerate = 0
                model.save()
            else:
                tolerate = tolerate+1

            if tolerate==limit:
                model.load()
                test_mse = wrapper_test(model)
                print('the best valid mse is:',str(best_mse))
                print('the test mse is ',str(test_mse))
                break


gpu_list = np.asarray(os.environ.get('CUDA_VISIBLE_DEVICES', '-1').split(','), dtype=np.int32)
args.n_gpu = len(gpu_list)
print('Initializing models')

model = Model(args)

if args.is_training == 1:
    wrapper_train(model)
elif args.is_training == 2:
    model.load()
    wrapper_use(model)
else:
    model.load()
    wrapper_test(model)