# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:11:37 2019

@author: jongg
"""

import numpy as np
import tensorflow as tf
#import scipy.misc1
from libs.environment import ENV_net
from libs.actor_critic_net import USL, USL_replay, Actor, Actor_USL, Critic, DDPG, PER, GaussianNoise
import scipy.io as sio
import os
from tqdm import tqdm
import glob



UE = 40
BS = 4
Bh = 20
DIR_UA = glob.glob(f'./Models/BS{BS}_UE{UE}_Bh{Bh}/UA_model*')
DIR_PC = glob.glob(f'./Models/BS{BS}_UE{UE}_Bh{Bh}/PC_model*')
save_file_UA  = f'{DIR_UA[0]}/UA_model'
save_file_PC = f'{DIR_PC[0]}/PC_model'


SNR_mat = sio.loadmat('./DATA/SNR_test.mat')['SNR'][:,:,:,0:BS]

env = ENV_net(SNR_mat, UE, BS, Bh)
Max_step = SNR_mat.shape[1]-1

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

PC_agent = DDPG(scope = 'PC_net',
                sess=sess,
                BS = BS,
                UE = UE,
                Actor = Actor(env.action_size_PC, thr = UE * BS, scope = 'PC_net_DDPG_Actor'),
                Critic = Critic(env.reward_size_PC, BS, UE, scope = 'PC_net_DDPG_Critic'),
                Actor_target = Actor(env.action_size_PC, thr= UE * BS,  scope = 'PC_net_DDPG_Actor_target'),
                Critic_target = Critic(env.reward_size_PC, BS, UE, scope = 'PC_net_DDPG_Critic_target'),
                OUNoise = GaussianNoise(env.action_size_PC,epsilon_init = 0.99,epsilon_end=0.7),
                replay_buffer = PER(buffer_size =100000, alpha = 0.7, epsilon_per = 0.1, beta = 0.4), 
                state_size = env.state_size_PC, 
                action_size = env.action_size_PC, 
                reward_size = env.reward_size_PC,
                gamma = 0.99, 
                lr_actor = 3e-7, 
                lr_critic = 1e-6, 
                batch_size = 512,
                tau = 0.001,
                is_tanh = True
                )

UA_agent = USL(scope = 'UA_net',
                sess= sess,
                BS = BS,
                UE = UE,
                Actor = Actor_USL(env.action_size_UA, scope = 'UA_net'),
                replay_buffer = USL_replay(buffer_size = 100000), 
                state_size = env.state_size_UA, 
                action_size = env.action_size_UA, 
                lr_actor = 3e-5, 
                batch_size = 2048,
                alpha_init= 0.01
                )



saver1 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='UA'))
saver1.restore(sess, save_file_UA)
saver2 = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='PC'))
saver2.restore(sess, save_file_PC)

data_indices = range(6)
block_size = 50
for data_index in data_indices:
    reward_list_PC = list()
    reward_list_UA = list()
    sr_list = list()
    qos_list = list()
    qos_list_amount = list()
    TD_error_PC = 0
    TD_error_UA = 0
    loss_list_UA = np.zeros([block_size])
    loss_list_PC = np.zeros([block_size])
    QoS_data = sio.loadmat(f'./DATA/save/BS{BS}_UE{UE}_Bh{Bh}_{data_index}.mat')['QoS']
    RES_data = sio.loadmat(f'./DATA/save/BS{BS}_UE{UE}_Bh{Bh}_{data_index}.mat')['RES']
    pbar = tqdm(range(block_size*data_index,block_size*(data_index+1) ))
    for env.episode in pbar :
        sr_wo_noise = list()
        reward_ep_PC = 0
        reward_ep_UA = 0
        env.step = 0
        env.reset(is_test=True)
        env.QoS = QoS_data[env.episode-data_index*block_size,:].reshape([-1,1])
        loss_set_PC = list()
        loss_set_UA = list()
        QoS_prob = 0
        QoS_amount = 0
        for env.step in range(Max_step):
            env.action = PC_agent.forward_noiseless_action(env.state_PC)            
            env.get_state_UA()
            env.action_UA = UA_agent.forward_action(env.state_UA)
            env.res_source = RES_data[env.episode-data_index*block_size,env.step,:]
            env.proceed_master()
            reward_ep_UA = reward_ep_UA +env.Lag
            loss_np = PC_agent.forward_loss(env.state_PC, env.state_next, env.action, env.reward)
            sr_wo_noise.append(env.sum_rate.copy())
            loss_set_PC.append(loss_np.copy())
            QoS_prob= QoS_prob + env.QoS_unsatisfactory
            QoS_amount = QoS_amount + np.sum(env.QoS_over_amount)
            reward_ep_PC = reward_ep_PC + env.reward
            env.state_PC = env.state_next.copy()
        qos_list.append(QoS_prob/Max_step)
        qos_list_amount.append(QoS_amount/Max_step)
        reward_list_UA.append(np.sum(reward_ep_UA)/Max_step)
        reward_list_PC.append(np.sum(reward_ep_PC[0,:])/Max_step)
        sr_list.append(np.mean(sr_wo_noise))
        loss_list_PC[env.episode-data_index*block_size] =np.mean(loss_set_PC)
        loss_list_UA[env.episode-data_index*block_size] =np.mean(loss_set_UA)
        pbar.set_postfix({'sr':np.mean(sr_list[-300::]), 'qos':np.mean(qos_list[-300::])})
    if not os.path.exists(f'./DATA/BS{BS}_UE{UE}_Bh{Bh}'):
        os.makedirs(f'./DATA/BS{BS}_UE{UE}_Bh{Bh}/')
    sio.savemat(f'./DATA/BS{BS}_UE{UE}_Bh{Bh}/fail_list_test_{data_index}.mat', {'data' : np.array(qos_list)})
    sio.savemat(f'./DATA/BS{BS}_UE{UE}_Bh{Bh}/fail_amount_list_test_{data_index}.mat', {'data' : np.array(qos_list_amount)})
    sio.savemat(f'./DATA/BS{BS}_UE{UE}_Bh{Bh}/sr_list_test_{data_index}.mat',{'data' : np.array(sr_list)})    

    
result_qos=0
result_sum_rate=0
for ind in range(6):
    avg_fail = np.mean(sio.loadmat(f'./DATA/BS{BS}_UE{UE}_Bh{Bh}/fail_list_test_{ind}.mat')['data'])
    avg_sr = np.mean(sio.loadmat(f'./DATA/BS{BS}_UE{UE}_Bh{Bh}/sr_list_test_{ind}.mat')['data'])
    result_qos += avg_fail
    result_sum_rate += avg_sr
    print(f'ind{ind}: SR - {avg_sr} , fail - {avg_fail}')
print(f'total: SR - {result_sum_rate/6} , fail - {result_qos/6}')

