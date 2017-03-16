# coding:utf-8


"""
data_proc.py

2017.1.13
速度追加

2017.1.8
load_proced_dataが返す値は
dict = {"joints":joints, "speaks":speaks, "annos":annos}
に変更

2016.12.8
複数人Joint&Annoを取得できるよう改良
user位置を固定(Y軸が小さい順にuser 0, 1, 2, ...)


2016.10.31
0のときのdataを捨てる関数作成cutting_data
 

Dataのとり方をTest
"""

import json
import math
import numpy as np


def load_data(datalen=-1, start=0, end=1000, select=True, normalize=True):
    pass

 
# 生Dataの入力とnetが読み込める形式に変換
def load_persons(filenames, datalen=-1, start=0, end=1000, annobool=True, select=True):#filename is list 
    ###---Entry Point---###

    # 1. Get N persons data from file(json)
    num = len(json.load(open(filenames[0], 'r')))
    joints_data = [[] for i in range(num)]
    speaks_data = [[] for i in range(num)]
    
    for filename in filenames:
        print "Filename:", filename     
        joints, speaks = json_load_data(filename, datalen, annobool) # e.g. (5000, 150),(2, 5000)
    
        # 2. Select joints index
        print "Select:", select
        if select==True:
            sidx = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 20, 25]
        else:
            sidx = np.arange(26)
            
        select_joints = []
        for jt in joints: 
            select_joints.append(select_data(jt, sidx)) # e.g. (5000, 36)
        
        # 3. Normalize data
        print "Normalize:", normalize
        users_joint, users_speak = [], []
        for jts, spks in zip(select_joints, speaks):
            norms = []
            orgy = 0
            for data in jts:
                pos, org = normalize_data(data, 3) 
                norms.append(pos)
                orgy += org[1]#人物位置のY座標
        
            # User 0, 1, ...の位置を決めるためorgを使う
            orgy_ave = orgy/len(jts) 
            #print "org_ave Y:", orgy_ave #y座標をみる

            users_joint.append([orgy_ave, norms])
            users_speak.append([orgy_ave, spks])

        #大きい順にsort
        users_joint_sort = sorted(users_joint, key=lambda x: float(x[0]))
        users_speak_sort = sorted(users_speak, key=lambda x: float(x[0]))

        for idx, (joint, speak) in enumerate(zip(users_joint_sort, users_speak_sort)):
            #print "val:",joint[0]
            joints_data[idx].extend(joint[1])
            speaks_data[idx].extend(speak[1])

        
    for jsd in joints_data:
        print len(jsd)
    
    dst_joints = np.hstack(joints_data[i] for i in range(len(joints_data)))
    dst_speaks = np.array(speaks_data).T

    dict = {"joints":dst_joints, "speaks":dst_speaks}
    
    return dict



# 生Dataの入力とnetが読み込める形式に変換
def load_data_persons(filenames, datalen=-1, start=0, end=1000, annobool=True, select=True, normalize=True):#filename is list 
    ###---Entry Point---###

    # 1. Get N persons data from file(json)
    num = len(json.load(open(filenames[0], 'r')))
    joints_data = [[] for i in range(num)]
    speaks_data = [[] for i in range(num)]
    
    for filename in filenames:
        print "Filename:", filename     
        joints, speaks = json_load_data(filename, datalen, annobool) # e.g. (5000, 150),(2, 5000)
    
        # 2. Select joints index
        print "Select:", select
        if select==True:
            sidx = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 20, 25]
        else:
            sidx = np.arange(26)
            
        select_joints = []
        for jt in joints: 
            select_joints.append(select_data(jt, sidx)) # e.g. (5000, 36)
        
        # 3. Normalize data
        print "Normalize:", normalize
        """
        if normalize==False:
            dst_joints = np.hstack(select_joints[i] for i in range(len(select_joints)))
            return dst_joints, dst_joints, speaks
        """

        users_joint, users_speak = [], []
        for jts, spks in zip(select_joints, speaks):
            norms = []
            orgy = 0
            for data in jts:
                pos, org = normalize_data(data, 3) 
                norms.append(pos)
                orgy += org[1]#人物位置のY座標
        
            # User 0, 1, ...の位置を決めるためorgを使う
            orgy_ave = orgy/len(jts) 
            #print "org_ave Y:", orgy_ave #y座標をみる

            users_joint.append([orgy_ave, norms])
            users_speak.append([orgy_ave, spks])

        #大きい順にsort
        users_joint_sort = sorted(users_joint, key=lambda x: float(x[0]))
        users_speak_sort = sorted(users_speak, key=lambda x: float(x[0]))

        for idx, (joint, speak) in enumerate(zip(users_joint_sort, users_speak_sort)):
            #print "val:",joint[0]
            joints_data[idx].extend(joint[1])
            speaks_data[idx].extend(speak[1])

        
    for jsd in joints_data:
        print len(jsd)
    
    dst_joints = np.hstack(joints_data[i] for i in range(len(joints_data)))
    dst_speaks = np.array(speaks_data).T
    
    return [dst_joints, dst_joints, dst_speaks]


# もしspeakがなければuser_speaksにすべて0をいれる
def json_load_data(filename, datalen, annobool):
    # User数を取得()

    f = open(filename, 'r')
    json_data = json.load(f)
    f.close()
    num = len(json_data)
    
    joints = []
    speaks = []

    # Userの数だけ
    for i in range(num): #user num
        jD = json_data[i]
        #JsonのKeyの変更に対応
        if jD.has_key("datas"):
            key_data, key_joints = "datas", "jdata"
        else:
            key_data, key_joints = "data", "joints"
                
        ds = len(jD[key_data])
                
        pose = [] #pose: shape(5000, 75)
        spks = [] #spks: shape(5000, 1)...0 or 1

        #dataを使う長さ指定
        uselen = len(jD[key_data]) if datalen == -1 else datalen
        print "uselen:", uselen
        for dl in range(uselen):

            user = jD[key_data][dl]
                
            ps = [] #ps: shape(75,)
            for ds in user[key_joints]:
                for p in ds:
                    ps.append(float(p)) #p: shape(3,)
            pose.append(ps)

            if user.has_key("speaked"):
                #閾値で切る
                if annobool:
                    spk = 1 if float(user["speaked"])>0.005 else 0
                else:
                    #decibel = float(user["speaked"]) 
                    spk = speakDecibelNorm(user["speaked"])
                    
                spks.append(spk)
            else:
                print "No speaked(alter 0)"
                spks.append(0)#speakedがなければ0

        # User i のdataを格納
        joints.append(pose)
        speaks.append(spks)

    dst_uj = np.array([joints[i] for i in range(num)], dtype=np.float32)
    dst_us = np.array([speaks[i] for i in range(num)], dtype=np.float32)

    #print dst_uj.shape, dst_us.shape
    
    return dst_uj, dst_us


def speakNorm(data, upper=300, lower=80):
    # 上限
    c_data=data if data <upper else upper
    # 下限
    c_data=c_data if c_data>lower else lower
    # 最大値で割る
    return (c_data-lower)/float(upper-lower)

def convDecibel(data):
    return 20*math.log10(data)

#decibel
def speakDecibelNorm(data):
    decibel = convDecibel(data)
    return speakNorm(decibel, upper=70, lower=30)

    
def select_data(data, idx):
    sidx = [sid*3+i  for sid in idx for i in range(3)]
    return data[:, sidx]

def select_data_online(data, idx):
    sidx = [sid*3+i  for sid in idx for i in range(3)]
    return data[sidx]

# e.g. d1:(5000,72), d2:(5000,72), spk:(5000,2), th: 10
def cutting_data(d1, d2, spk, th):
    
    flag = np.sum(spk, axis=1)            
    cd1, cd2, cspk = [], [], []
    for fg, data1, data2, speak in zip(flag, d1, d2, spk):
        if fg == 1:
            cd1.append(data1)
            cd2.append(data2)
            cspk.append(speak)
        
    #print cd1.shape, cd2.shape, cspk.shape
    return np.array(cd1, dtype=np.float32), np.array(cd2, dtype=np.float32), np.array(cspk, dtype=np.float32)
        
    
# data.shape (N,)
def normalize_data(data, os):
    # 原点の計算
    def calc_org(data, s_l_id, s_r_id, spn_id, os):
        #print data
        s_l, s_r, spn = data[s_l_id], data[s_r_id], data[spn_id] 
        a, b = 0, 0
        # 原点 = 右肩から左肩にかけた辺と胸からの垂線の交点
        for i in range(os):
            a += (spn[i]-s_l[i])*(s_r[i]-s_l[i])
            b += (s_r[i]-s_l[i])**2
        k = a/b
        return np.array([k*s_r[i]+(1-k)*s_l[i] for i in range(os)])

    # 法線の計算(from spn)
    def calc_normal(d_sl, d_sp, d_sr):
        l_s = np.array(d_sl) - np.array(d_sp)
        s_r = np.array(d_sp) - np.array(d_sr)
        x = l_s[1]*s_r[2]-l_s[2]*s_r[1]
        y = l_s[2]*s_r[0]-l_s[0]*s_r[2]
        z = l_s[0]*s_r[1]-l_s[1]*s_r[0]
        return np.array([x, y, z])

    # 回転行列による変換
    def calc_rot_pose(data, th_z, th_y, org):
        cos_th_z, sin_th_z = np.cos(-th_z), np.sin(-th_z) 
        cos_th_y, sin_th_y = np.cos(th_y), np.sin(th_y)
        rot1 = np.array([[cos_th_z, -sin_th_z, 0],[sin_th_z, cos_th_z, 0],[0, 0, 1]])
        rot2 = np.array([[cos_th_y, 0, sin_th_y],[0, 1, 0],[-sin_th_y, 0, cos_th_y]])
        rot_pose = []
        for dt in data:
            dt = np.array(dt)-org
            rd = np.dot(rot1, dt)
            rd = np.dot(rot2, rd)
            rd = rd+org
            rot_pose.append(rd)
        return rot_pose

    # 平行移動
    def calc_shift_pose(data, org, s):
        #print org.shape
        shift = s - org
        shift_pose = []
        for dt in data:
            dt += shift
            shift_pose.append(dt)
        return np.array(shift_pose)

    def data_set(data, os):
        ds = np.reshape(data, (len(data)/os, os))
        return ds

    def data_reset(data):
        data = np.array(data, dtype=np.float32) # dtype is important!
        ds = np.reshape(data, (data.shape[0]*data.shape[1], ))
        return ds
        
    #[1,2,3,...] -> [[1,2,3],[...],...]
    data = data_set(data, os)
    
    # 左肩 4, 右肩 7(raw joints was cutted), 胸 1 
    s_l_id, s_r_id, spn_id = 4, 7, 1
    
    # 原点の計算
    org = calc_org(data, s_l_id, s_r_id, spn_id, os)
    # 法線を求める
    normal = calc_normal(data[spn_id], org, data[s_l_id])
    #normal = calc_normal(data[s_l_id], data[spn_id], data[s_r_id])
    
    # 法線の角度方向にposeを回転
    th_z = np.arctan2(normal[1], normal[0])-np.arctan2(org[1], org[0]) #z軸回転(法線と原点の間の角度)
    th_y = np.arctan2(normal[2], normal[0])-np.arctan2(org[2], org[0]) #y軸回転 
    rot_pose = calc_rot_pose(data, th_z, th_y, org)
    #orgをx軸上に変換する
    th_z = np.arctan2(org[1], org[0])
    th_y = np.arctan2(org[2], org[0])
    rot_pose_norm = calc_rot_pose(rot_pose, th_z, th_y, np.array([0,0,0]))
    # 変換後のorg
    rot_org = calc_org(rot_pose_norm, s_l_id, s_r_id, spn_id, os)
    # orgのxを特定の値に移動する(origen)
    s = [0,0,0]
    shift_pose = calc_shift_pose(rot_pose_norm, rot_org, s)
    shift_org = calc_org(shift_pose, s_l_id, s_r_id, spn_id, os)
        
    return data_reset(shift_pose), org


def save_data(name_json, keys, data):
    dict = {"proced": 1,}
    for k, d in zip(keys, data):
        dict[k] = d.tolist()
        
    open(name_json, 'w').write(json.dumps(dict))
    
    print "save:", name_json
    
#処理済のJSONfileを入力
def load_proced_data(filename, datalen=-1, start=0, end=0):
    print filename
    if len(filename) == 1:
        #Fileが一つのとき
        print "filename:", filename
        json_file = json.load(open(filename[0], 'r'))
        if datalen == -1:
            if start < end:
                joints = np.array(json_file["joints"], dtype=np.float32)[start:end]
                speaks = np.array(json_file["speaks"], dtype=np.float32)[start:end]
                annos = np.array(json_file["annos"], dtype=np.float32)[start:end]
            else:
                joints = np.array(json_file["joints"], dtype=np.float32)
                speaks = np.array(json_file["speaks"], dtype=np.float32)
                annos = np.array(json_file["annos"], dtype=np.float32)
        else:
            joints = np.array(json_file["joints"], dtype=np.float32)[:datalen]
            speaks = np.array(json_file["speaks"], dtype=np.float32)[:datalen]
            annos = np.array(json_file["annos"], dtype=np.float32)[:datalen]
        return [joints, speaks, annos]    
    else:
        #Fileが複数のとき
        for i, fname in enumerate(filename):
            print "filename", i, fname
            json_file = json.load(open(fname, 'r'))
            if i == 0:
                joints = np.array(json_file["joints"], dtype=np.float32)
                speaks = np.array(json_file["speaks"], dtype=np.float32)
                annos = np.array(json_file["annos"], dtype=np.float32)
            else:
                joints = np.vstack((joints, np.array(json_file["joints"], dtype=np.float32)))
                speaks = np.vstack((speaks, np.array(json_file["speaks"], dtype=np.float32)))
                annos = np.vstack((annos, np.array(json_file["annos"], dtype=np.float32)))              
        if datalen != -1:
            if start < end:
                joints = joints[start:end]
                speaks = speaks[start:end]
                annos = annos[start:end]
            else:
                joints = joints[:datalen]
                speaks = speaks[:datalen]
                annos = annos[:datalen]
                
        return [joints, speaks, annos]


def load_proced_data_flag(filename, datalen=-1, start=0, end=0):

    for i, fname in enumerate(filename):
        print "filename", i, fname
        json_file = json.load(open(fname, 'r'))
        if i == 0:
            joints = np.array(json_file["joints"], dtype=np.float32)
            speaks = np.array(json_file["speaks"], dtype=np.float32)
            annos = np.array(json_file["annos"], dtype=np.float32)
            if json_file.has_key("flags"): 
                flags = np.array(json_file["flags"], dtype=np.float32)
            else:
                flags = np.zeros((annos.shape[0], 1))
        else:
            joints = np.vstack((joints, np.array(json_file["joints"], dtype=np.float32)))
            speaks = np.vstack((speaks, np.array(json_file["speaks"], dtype=np.float32)))
            annos = np.vstack((annos, np.array(json_file["annos"], dtype=np.float32)))
            if json_file.has_key("flags"):
                flags = np.vstack((flags, np.array(json_file["flags"], dtype=np.float32)))
            else:
                flags = np.vstack((flags, np.zeros((annos.shape[0], 1))))
            
    if datalen != -1: 
        if start < end:
            joints = joints[start:end]
            speaks = speaks[start:end]
            annos = annos[start:end]
            flags = flags[start:end] 
        else:
            joints = joints[:datalen]
            speaks = speaks[:datalen]
            annos = annos[:datalen]
            flags = flags[:datalen]

    dict = {"joints":joints, "speaks":speaks, "annos":annos, "flags":flags}
    
    return dict
    
    pass

"""
def load_proced_data(filename, datalen=-1):
    print filename
    if len(filename)==1:
        print "filename:", filename
        json_file = json.load(open(filename[0], 'r'))

        #JsonのKeyの変更に対応
        if json_file.has_key("annos"):
            if datalen == -1:
                joints = np.array(json_file["joints"], dtype=np.float32)
                speaks = np.array(json_file["speaks"], dtype=np.float32)
                annos = np.array(json_file["annos"], dtype=np.float32)
            else:
                joints = np.array(json_file["joints"], dtype=np.float32)[:datalen]
                speaks = np.array(json_file["speaks"], dtype=np.float32)[:datalen]
                speaks = np.array(json_file["annos"], dtype=np.float32)[:datalen]
            return [joints, speaks, annos]
        else:
            if datalen == -1:
                joints = np.array(json_file["joints"], dtype=np.float32)
                speaks = np.array(json_file["speaks"], dtype=np.float32)
            else:
                joints = np.array(json_file["joints"], dtype=np.float32)[:datalen]
                speaks = np.array(json_file["speaks"], dtype=np.float32)[:datalen]
    
            return [joints, speaks]

    else:      
        for i, fname in enumerate(filename):
            print "filename", i, fname
            json_file = json.load(open(fname, 'r'))

            if i == 0:
                joints = np.array(json_file["joints"], dtype=np.float32)
                speaks = np.array(json_file["speaks"], dtype=np.float32)
            else:
                joints = np.vstack((joints, np.array(json_file["joints"], dtype=np.float32)))
                speaks = np.vstack((speaks, np.array(json_file["speaks"], dtype=np.float32)))
                
        if datalen != -1:
            joints = joints[:datalen]
            speaks = speaks[:datalen]

        return [joints, speaks]
"""  
    
# annotationの処理
def proc_anno(joints, speak, use_vote=False, use_speak=True, threshold=-0.2):
    
    # 腕の高さ判定
    def joint_z_th(data, th):
        return [1 if d > th else 0 for d in data]

    # 腕の位置判定
    def joint_x_th(data, th):
        return [1 if d > th else 0 for d in data]

    
    # 多数決Filter 隙間の穴埋め
    def vote_sig(data, win):
        vt = []
        vt.extend(data[1:1+win][::-1])
        vt.extend(data)
        vt.extend(data[-(win+1):-1][::-1])
        return [round(sum(vt[i:i+2*win+1])/float(len(vt[i:i+2*win+1]))) for i in range(len(vt)-2*win)]

    # Entry Point
    print "proc anno (use_vote:", use_vote, ", use_speak:", use_speak, ")"
    th = threshold
    user_num = speak.shape[1] #temp user num
    joint_num = 12# ALL Joints Num
    wrists = np.array([6, 9]) # 左右の手首のIndex
    idxs = [wrists+joint_num*i for i in range(user_num)] #user分の手首のindex Shape: e.g. (3, 2)
    
    #z座標を取り出す(joints[:,i*3+2])
    z_sigs = [[joint_z_th(joints[:,i*3+2], th) for i in  idx] for idx in idxs] # Shape: e.g. (3, 2, 5000)

    #x座標を取り出す(joints[:,i*3+2])
    #z_sigs = [[joint_x_th(joints[:,i*3], th) for i in  idx] for idx in idxs] # Shape: e.g. (3, 2, 5000)

    
    user_annos = []
    for sigs, spks in zip(z_sigs, speak.T): # e.g. z_sigs:(3, 2, 5000), speak.T:(3, 5000)
        sigs = np.array(sigs)
        anns = []
        for sig, spk in zip(sigs.T, spks): # e.g. sigs.T:(5000, 2), spks:(5000,)
            if use_speak:
                ann = 1 if sig[0]+sig[1]!=0 and spk==True else 0 #そのspeak annoが正当かどうか
            else:
                ann = 1 if sig[0]+sig[1]!=0 else 0 #動きだけでAnnotationする
            anns.append(ann)
        user_annos.append(anns) #(3, 5000)
         
    if use_vote == False:
        return np.array(user_annos, dtype=np.float32).T

    win = 30
    vote_user_anns = [vote_sig(ann, win) for ann in user_annos]
    return np.array(vote_user_anns, dtype=np.float32).T


#速度計算
def calc_velocity_online(now, past):
    diff = now - past #(24, 3)
    innpro = np.dot(diff, diff.T)
    return  np.sqrt(np.diag(innpro))#(1, 24)


#user1, user2のt,t-1の差分のnormを計算
def calc_velocity_from_dataset(joints, sidx=[]):
    
    #joints(1000, 72)
    #datalen = joints.shape[0]

    length, dim = joints.shape
    vels = []
    offset = 3
    
    if dim%offset != 0:
        print "bad shape:", joints.shape
        
    for i in range(length):

        if i == 0:
            # 初期値は0
            vel = np.zeros(dim/offset)#(24,)
        else:
            now = joints[i].reshape(dim/offset, offset) #(24, 3)
            past =joints[i-1].reshape(dim/offset, offset)#(24, 3)
            vel = calc_velocity_online(now, past)

        if len(sidx)>0:
            vel = vel[sidx]
            
        vels.append(vel)
        
    return np.array(vels).astype(np.float32)


#user1, user2のt,t-1の差分のnormを計算
def calc_velocities_from_dataset(joints, sidx=[]):
    
    #joints(1000, 72)
    #datalen = joints.shape[0]

    length, dim = joints.shape
    vels = []
    offset = 3
    
    if dim%offset != 0:
        print "bad shape:", joints.shape
        
    for i in range(length):

        if i == 0:
            # 初期値は0
            vel = np.zeros(dim)#(36,)
        else:
            now = joints[i]
            past =joints[i-1]
            vel = now-past

        if len(sidx)>0:
            #data cut
            vel = vel[sidx]
            
        vels.append(vel)
        
    return np.array(vels).astype(np.float32)
