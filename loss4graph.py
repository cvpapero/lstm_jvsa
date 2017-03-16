#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--folder', '-f', default="",  
                    help='folder name')
parser.add_argument('--mse', '-m', default=True,  
                    help='rep(m)')
parser.add_argument('--epoch', '-e', default=-1, type=int, 
                    help='visualize epoch')
args = parser.parse_args()


#print "viz loss:", args.folder 
foldernames = [args.folder]

filenames = []
paramnames = []
for folder in foldernames:
    filenames.append(folder+"/loss_data.json")
    paramnames.append(folder+"/param.json")

fs = 10
    
labelnames = []
for param in paramnames:
    data = json.load(open(param))
    if ("version" in data) == False:
        print "save version 1.0"
        label = str(data["network_name"])+"_layer:"+str(data["n_layers"])+"_unit:"+str(data["dim_h"])
        labelnames.append(label)
    else:
        print "save version", data["version"]
        label = str(data["net"]["name"])+"_layer:"+str(data["net"]["n_layers"])+"_unit:"+str(data["net"]["dim_h"])
        labelnames.append(label)
    
# lossの取得
loss_array = []
s_loss_array = []



for file in filenames:
    data = json.load(open(file))
    loss_data = data["loss"]#(1000,5)
    s_loss_data = data["s_loss"]#(1000,5)
    #valid_loss_data = data["val_loss"]
    if args.mse != True:
        print "not mse!"
        loss_data = np.sqrt(np.array(loss_data))
        s_loss_data = np.sqrt(np.array(s_loss_data))
        #valid_loss_data = np.sqrt(np.array(valid_loss_data))
    loss_array.append(loss_data)
    s_loss_array.append(s_loss_data)
    #valid_loss_array.append(valid_loss_data)

    
loss_array = np.array(loss_array)
s_loss_array = np.array(s_loss_array)

#valid_loss_array = np.array(valid_loss_array)

length = loss_array.shape[1]
if args.epoch != -1:
    length = args.epoch
    

# lossの可視化
print "loss_array:", loss_array.shape
labelset = ["train", "s_train"]
#styleset = [[2,1,1], [2,4,5], [2,4,6], [2,4,7], [2,4,8]]
#styleset = [[5,1,1], [5,1,2], [5,1,3], [5,1,4], [5,1,5]]
styleset = [[3,1,1], [3,2,3], [3,2,4], [3,2,5], [3,2,6]]


titleset = ["total", "position", "difference", "voice", "speaker"]
for (i, label) in enumerate(labelnames):
    # Total
    for j in range(len(titleset)):
        plt.subplot(styleset[j][0], styleset[j][1], styleset[j][2])
        plt.plot(loss_array[i,:,j], label=labelset[0])
        #plt.plot(s_loss_array[i,:,j], label=labelset[1])
        #plt.plot(loss_array[i,:length,j], label=labelset[0])
        #plt.plot(valid_loss_array[i,:length,j], label=labelset[1])
        #plt.legend(fontsize=fs+1)
        last_loss = round(loss_array[i,length-1,j],4)
        #s_last_loss = round(s_loss_array[i,length-1,j],4)
        title_set = titleset[j]
        print titleset[j]+": epoch_"+str(length)+", loss_"+str(last_loss)#+", s_loss_"+str(s_last_loss) 
        plt.title(title_set, fontsize=fs+2)
        plt.xlim(0,length)
        #plt.tick_params(labelsize=fs)
        #plt.xlabel("epoch", fontsize=fs)

        
        if args.mse == True:
            plt.ylabel("MSE", fontsize=fs)
        else:
            plt.ylabel("root MSE", fontsize=fs)
        
    """
    # Joints
    plt.subplot(2, 2, 3)
    plt.plot(loss_array[i,:,1], label=label_train)
    plt.plot(valid_loss_array[i,:,1], label=label_valid)

    # Annotation
    plt.subplot(2, 2, 4)
    plt.plot(loss_array[i,:,2], label=label_train)
    plt.plot(valid_loss_array[i,:,2], label=label_valid)
    #plt.legend(fontsize=fs)
    #plt.grid(which='major',color='black',linestyle='-')
    #plt.grid(which='minor',color='black',linestyle='-')
    """
    print "input:", label
    print "min[", loss_array[i].argmin(),"]:",loss_array[i].min()
    print "last epoch:", loss_array[i,-1]

    

"""
plt.tick_params(labelsize=fs)

plt.xlabel("epoch", fontsize=fs)
if args.mse == True:
    plt.ylabel("mean squared error", fontsize=fs)
else:
    plt.ylabel("[m]", fontsize=fs)
"""

plt.tight_layout()
    
plt.show()
