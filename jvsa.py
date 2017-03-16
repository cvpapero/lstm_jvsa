# coding:utf-8

"""
2017.1.16
使用する関節角度を指定する(全部はいらない)

2017.1.13
特定のseqは個別にBPTT

velを追加

2017.1.11
joint+speakひとつのvectorにする, 
annotation



2017.1.9
rnn_annotation2_1.pyを改良
DataのInputにおいてdata_procを使う
-procされたDataしか使わない
"""

import argparse
import copy
from datetime import datetime 
import glob
import time
import sys
import json
import numpy as np
import six
import os
import tqdm

import net
import data_proc2

import chainer
from chainer import cuda
import chainer.links as L
import chainer.functions as F
from chainer import optimizers
from chainer import serializers

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', '-g', default=-1, type=int,
                    help='GPU ID (negative value indicates CPU)')

parser.add_argument('--initmodel', '-md', default="",
                    help='init model')
parser.add_argument('--resume', '-op', default="",
                    help='optimizer')
parser.add_argument('--usemodel', '-um', default="LSTM_ANN",
                    help='use model')
parser.add_argument('--n_units', '-ut', default=50, type=int,
                    help='layers')
parser.add_argument('--n_layers', '-ly', default=2, type=int,
                    help='layers')
parser.add_argument('--n_epoch', '-ep', default=500, type=int,
                    help='epoch')
parser.add_argument('--batchsize', '-bt', default=20, type=int,
                    help='batch size')
parser.add_argument('--bprop_len', '-bl', default=50, type=int,
                    help='backprop length')

parser.add_argument('--train_dir', '-trd', default='proced_20170107/3seqx2_raw-spk-40-60',
                    help='filename')

parser.add_argument('--s_train_dir', '-strd', default='proced_20170107/nod',
                    help='filename')

parser.add_argument('--valid_dir', '-vld', default='proced_20170107/train',
                    help='filename')
parser.add_argument('--test_dir', '-tsd', default='proced_20170107/train',
                    help='filename')

parser.add_argument('--datalen', '-dl', default=3000, type=int,
                    help='-1:no cut')
parser.add_argument('--note', '-nt', default="",
                    help='description of this network')

args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np
if args.gpu >= 0:
    print "GPU ON"

    
def mkdir(path):
    os.mkdir(path)
    print "Folder [", path, "] is created."
    

def evaluate(dataset, annoset):
    evaluator = model.copy()  # to use different state
    evaluator.reset_state()  # initialize state
    sum_loss, sum_loss_x, sum_loss_a = 0, 0, 0
    data_len = dataset[0].shape[0]
    for i in six.moves.range(data_len - 1):
        #print i
        x = chainer.Variable(xp.asarray(dataset[0][i:i + 1]), volatile='on')
        x_t = chainer.Variable(xp.asarray(dataset[1][i + 1:i + 2]), volatile='on')

        a = chainer.Variable(xp.asarray(annoset[0][i:i + 1]), volatile='on')
        a_t = chainer.Variable(xp.asarray(annoset[1][i + 1:i + 2]), volatile='on')
        
        loss, loss_x, loss_a = evaluator(x, a, x_t, a_t)
        
        sum_loss += loss.data
        sum_loss_x += loss_x.data
        sum_loss_a += loss_a.data
        
    return [float(sum_loss)/(data_len-1), float(sum_loss_x)/(data_len-1), float(sum_loss_a)/(data_len-1)]



# ---Entry Point---



n_layers = args.n_layers
n_epoch = args.n_epoch
dim_h = args.n_units
batchsize = args.batchsize  
bprop_len = args.bprop_len
grad_clip = 1

train_file = glob.glob(args.train_dir+"/*")
s_train_file = glob.glob(args.s_train_dir+"/*")

valid_file = glob.glob(args.valid_dir+"/*")
test_file = glob.glob(args.test_dir+"/*")

if len(train_file) == 0:
    print "error train folder no file!"

train_data = data_proc2.load_proced_data(train_file) #(joints, speaks, annos)
s_train_data = data_proc2.load_proced_data_flag(s_train_file) #(joints, speaks, annos)

valid_data = data_proc2.load_proced_data(valid_file, datalen=args.datalen)
test_data = data_proc2.load_proced_data(test_file, datalen=args.datalen)

print("train_data:",train_data[0].shape,train_data[2].shape,train_data[2].shape) 
#print("valid_data[0]:",valid_data[0].shape)
#print("test_data[0]:",test_data[0].shape)

# Data set
train_joints, train_speaks, train_annos = train_data[0], train_data[1], train_data[2]
s_train_joints, s_train_speaks, s_train_annos, s_train_flags = s_train_data["joints"], s_train_data["speaks"], s_train_data["annos"], s_train_data["flags"]

valid_joints, valid_speaks, valid_annos = valid_data[0], valid_data[1], valid_data[2]
test_joints, test_speaks, test_annos = test_data[0], test_data[1], test_data[2]

# 速度計算
#sidx = [6, 9, 11, 6+12, 9+12, 11+12]
sidx=[]
train_vels = data_proc2.calc_velocity_from_dataset(train_joints, sidx=sidx)
s_train_vels = data_proc2.calc_velocity_from_dataset(s_train_joints, sidx=sidx)

print "create velocity data:", train_vels.shape

dim_j = train_joints.shape[1]
dim_v = train_vels.shape[1]
dim_s = train_speaks.shape[1]
dim_a = train_annos.shape[1]

#import matplotlib.pyplot as plt
#plt.plot(train_vels)
#plt.show()
#sys.exit()


#joint+speak, annotation
model = net.LSTM_ANN(dim_j+dim_v+dim_s, dim_a, dim_h)

for param in model.params():
    data = param.data
    data[:] = np.random.uniform(-0.1, 0.1, data.shape)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()
    

# Setup optimizer
#optimizer = optimizers.SGD(lr=1.)
optimizer = optimizers.Adam()
optimizer.setup(model)
#optimizer.add_hook(chainer.optimizer.GradientClipping(grad_clip))

"""
# Init/Resume
if args.initmodel:
    print('Load model from', args.initmodel)
    serializers.load_npz(args.initmodel, model)
if args.resume:
    print('Load optimizer state from', args.resume)
    serializers.load_npz(args.resume, optimizer)
"""  

# Learning loop
whole_len = train_data[0].shape[0]  #train_data[0] is shape(5000, 66)
jump = whole_len // batchsize    # バッチサイズずつに分割した場合の幅

save_base = jump * n_epoch // 10
print "save_base",save_base

network_name = str(model.__class__.__name__)
folder_name = network_name+"_"+datetime.now().strftime("%Y%m%d_%H%M%S")
mkdir(folder_name)

dict = {
    "net":{
        "name":network_name,
        "model":network_name+"_final",
        "initmodel":args.initmodel,
        "optimizer":args.resume,
        "dim_j":dim_j,
        "dim_v":dim_v,
        "dim_s":dim_s,        
        "dim_a":dim_a,
        "dim_h":dim_h,
        "n_layers":n_layers,
    },
    "data":{
        "train":{"name":train_file,
                 "shape":{
                     "joints": train_joints.shape,
                     "speaks": train_speaks.shape,
                     "annos": train_annos.shape,
                 },
        },
        "second_train":{"name":s_train_file,
                 "shape":{
                     "joints": s_train_joints.shape,
                     "speaks": s_train_speaks.shape,
                     "annos": s_train_annos.shape,
                 },
        },
        "valid":{"name":valid_file,
                 "shape":{
                     "joints": valid_joints.shape,
                     "speaks": valid_speaks.shape,
                     "annos": valid_annos.shape,
                 },
        },
        "test":{"name":test_file,
                "shape":{
                     "joints": test_joints.shape,
                     "speaks": test_speaks.shape,
                     "annos": test_annos.shape,
                 },
        },
    },
    "other": {
        "n_epoch":n_epoch,
        "batchsize":batchsize,  
        "bprop_len":bprop_len,
        "whole_len":whole_len,
        "jump":jump,    
        "save_base":save_base,
    },
    "note":args.note,
    "code_name":__file__,
    "version":2.0,
}


open(folder_name+"/param.json", 'w').write(json.dumps(dict, indent=4))

accum_loss = 0
loss_data = []
val_loss_data = []
cur_loss, cur_loss_j, cur_loss_s, cur_loss_a = xp.zeros(()), xp.zeros(()), xp.zeros(()), xp.zeros(())

s_loss_data = []
s_val_loss_data = []
        
start_at = time.time()
cur_at = start_at
start_time = time.time()

epoch = 0

batch_idxs = list(range(batchsize))

print('Going to train {} iterations'.format(jump * n_epoch))
print "jump",jump

for i in six.moves.range(jump * n_epoch):
    #print i, [(jump * j + i) % whole_len for j in batch_idxs]
    #Joint
    jnt_data = xp.asarray([train_joints[(jump * j + i) % whole_len,:] for j in batch_idxs])#(20,72)
    jnt_label = xp.asarray([train_joints[(jump * j + i + 1) % whole_len,:] for j in batch_idxs])

    #Velocity
    vel_data = xp.asarray([train_vels[(jump * j + i) % whole_len,:] for j in batch_idxs])#(20,24)
    vel_label = xp.asarray([train_vels[(jump * j + i + 1) % whole_len,:] for j in batch_idxs])
    
    # Speak
    spk_data = xp.asarray([train_annos[(jump * j + i) % whole_len] for j in batch_idxs]).reshape(len(batch_idxs), dim_s)#(20,2)
    spk_label = xp.asarray([train_annos[(jump * j + i + 1) % whole_len] for j in batch_idxs]).reshape(len(batch_idxs), dim_s)

    
    jnt_data = np.hstack((jnt_data, vel_data, spk_data))#(20,98)
    jnt_label = np.hstack((jnt_label, vel_label, spk_label))

    jnt = chainer.Variable(jnt_data)
    jnt_t = chainer.Variable(jnt_label)

    
    # Annotation 
    ann_data = xp.asarray([train_annos[(jump * j + i) % whole_len] for j in batch_idxs]).reshape(len(batch_idxs), dim_a)
    ann_label = xp.asarray([train_annos[(jump * j + i + 1) % whole_len] for j in batch_idxs]).reshape(len(batch_idxs), dim_a)
    ann = chainer.Variable(ann_data)
    ann_t = chainer.Variable(ann_label)

    #print j, s, a
    
    loss_i, loss_i_j, loss_i_a = model(jnt, ann, jnt_t, ann_t)#(loss, loss_j, loss_s, loss_a)
    accum_loss += loss_i # 累計損失を計算
    
    cur_loss += loss_i.data
    cur_loss_j += loss_i_j.data
    #cur_loss_s += loss_i_s.data
    cur_loss_a += loss_i_a.data

    # Modelの学習
    if (i + 1) % bprop_len == 0:  # Run truncated BPTT
        model.zerograds()  # 勾配をゼロ初期化
        accum_loss.backward() # 累計損失を使って、誤差逆伝播(誤差の計算)
        accum_loss.unchain_backward()  # truncate # 誤差逆伝播した変数や関数へのreferenceを削除
        accum_loss = 0
        optimizer.update() # 最適化ルーチンの実行

    
    # Modelの学習2
    if (i + 1) % jump == 0: 
        s_data_len=s_train_joints.shape[0]
        s_accum_loss = 0
        s_cur_loss, s_cur_loss_j, s_cur_loss_a = xp.zeros(()), xp.zeros(()), xp.zeros(())
        for s_i in six.moves.range(s_data_len - 1):
            s_jnt, s_jnt_t = s_train_joints[s_i:s_i+1,:], s_train_joints[s_i+1:s_i+2,:]
            s_vel, s_vel_t = s_train_vels[s_i:s_i+1,:], s_train_vels[s_i+1:s_i+2,:]
            s_spk, s_spk_t = s_train_speaks[s_i:s_i+1,:], s_train_speaks[s_i+1:s_i+2,:]
            s_ann, s_ann_t = s_train_annos[s_i:s_i+1,:], s_train_annos[s_i+1:s_i+2,:]
            
            s_jnt = chainer.Variable(np.hstack((s_jnt, s_vel, s_spk)))#(1,74)
            s_jnt_t = chainer.Variable(np.hstack((s_jnt_t, s_vel_t, s_spk_t)))
        
            s_ann = chainer.Variable(s_ann)
            s_ann_t = chainer.Variable(s_ann_t)

            s_loss_i, s_loss_i_j, s_loss_i_a = model(s_jnt, s_ann, s_jnt_t, s_ann_t)
            s_accum_loss += s_loss_i # 累計損失を計算
            
            s_cur_loss += s_loss_i.data
            s_cur_loss_j += s_loss_i_j.data
            #s_cur_loss += s_loss_i_s.data
            s_cur_loss_a += s_loss_i_a.data
            
        
            if s_train_flags[s_i+1] == 1: #末尾がズレてる
                #print "bptt",s_i
                #print anno_st1
                #print anno_st2
                #anno_st1,anno_st2=[],[]
                model.reset_state()
                model.zerograds()  # 勾配をゼロ初期化
                s_accum_loss.backward() # 累計損失を使って、誤差逆伝播(誤差の計算)
                s_accum_loss.unchain_backward()  # truncate # 誤差逆伝播した変数や関数へのreferenceを削除
                s_accum_loss = 0
                optimizer.update() # 最適化ルーチンの実行

        s_loss = [s_cur_loss/float(s_data_len-1), s_cur_loss_j/float(s_data_len-1), s_cur_loss_a/float(s_data_len-1)]
        s_cur_loss.fill(0)
        s_cur_loss_j.fill(0)
        #cur_loss_s.fill(0)
        s_cur_loss_a.fill(0)
        print "---second train_loss:", s_loss
        model.reset_state()#重要...外すとbatchsizeが間違ってると怒られる
    
    # Modelの評価
    if (i + 1) % jump == 0:       
        epoch += 1        
        print "epoch:", epoch, ", iter:", i+1
        now = time.time()
        #loss = [cur_loss/float(jump), cur_loss_j/float(jump), cur_loss_s/float(jump), cur_loss_a/float(jump)]
        loss = [cur_loss/float(jump), cur_loss_j/float(jump), cur_loss_a/float(jump)]
        cur_loss.fill(0)
        cur_loss_j.fill(0)
        #cur_loss_s.fill(0)
        cur_loss_a.fill(0)
        print "---train_loss:", loss
        loss_data.append(loss)

        """
        val_loss = evaluate(valid_joints, valid_speaks, valid_annos)    
        print "---valid_loss:", val_loss
        val_loss_data.append(val_loss)
        cur_at += time.time() - now  # skip time of evaluation     
        """
        
    # Modelの保存
    if (i+1) % save_base == 0:
        save_model = copy.deepcopy(model)
        if args.gpu >0:
            save_model.to_cpu()
        serializers.save_npz(folder_name+'/'+network_name+'_ep'+str(epoch)+'.model', save_model)
        print 'save rnnlm_proc_'+str(epoch)+'.model'
        
    sys.stdout.flush()
    
finish_time = time.time()
elapsed_time = finish_time - start_time
print "elapsed time[sec]:", elapsed_time
#print int(elapsed_time/(60**2)),":", int(elapsed_time%(60**2)/60), ":",(elapsed_time%60)/60
# Save Loss 
dict = {
    "loss":loss_data,
    "val_loss":val_loss_data,
    "elapsed_time":elapsed_time,
}
open(folder_name+"/loss_data.json", 'w').write(json.dumps(dict, indent=4))
    

# Save the model and the optimizer
print "save the model"
if args.gpu > 0:
    model.to_cpu()
serializers.save_npz(folder_name+'/'+network_name+'_final.model', model)

print "save the optimizer"
serializers.save_npz(folder_name+'/'+network_name+'_final.state', optimizer)

# Evaluate on test dataset
#test_loss = evaluate(test_joints, test_speaks, test_annos)
#print "test loss:", test_loss






