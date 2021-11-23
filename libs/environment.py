import numpy as np 

class ENV_net():
    def __init__(self,SNR_mat, UE, BS, Bh):
        self.UE = UE
        self.BS = BS
        self.episode = 0
        self.step = 0
        self.max_level = 500
        self.power_default = 37
        self.renew_max = 60
        self.renew_min = 37
        self.tx_dBm = 30
        self.tx_w  = np.power(10, self.tx_dBm/10)/1000
        self.delta = 2.6
        self.grid_power_min = 200
        self.grid_power_max = 200
        self.QoS_pool = np.array([0.192,2.22,1.5,0.6,4.44]) ## Mbps [Audio, Video, Image, Web_bro, Email]
        self.SNR_mat = SNR_mat
        self.renew = (np.arange(self.BS) != self.BS-1).astype(int) #np.array([1,1,1,0])
        self.grid_power = self.grid_power_min + (self.grid_power_max-self.grid_power_min)*np.random.uniform(size= [self.BS])
        self.grid_power = self.grid_power * (1-self.renew)
        self.Backhaul_lim = Bh
        self.Backhaul = self.Backhaul_lim  + (50-self.Backhaul_lim )*(1-self.renew)
        self.RB = 100
        self.BW = 2e+7
        self.action_size_PC  = self.BS 
        self.action_size_UA = 2*self.BS
        self.reward_size_PC =  2 * self.BS #3*self.BS
        
        self.reward_size_UA = 4*self.BS #+ 2*self.UE
        self.state_size_PC  = self.UE*self.BS + 5*self.BS + self.UE
        self.state_size_UA = self.UE*self.BS + 2* self.BS + self.UE
        self.QoS_penalty = 4.0
        self.Backhaul_penalty =100

    def reset(self,is_test=False):
        
        if is_test:
            self.UA_set = np.arange(self.UE)
            self.H_mat = self.SNR_mat[self.episode, :, :,:].copy()
        else:
            self.UA_set = np.random.permutation(self.SNR_mat.shape[2])[0:self.UE]
            self.H_mat = self.SNR_mat[np.mod(self.episode+int(np.random.uniform(0,self.SNR_mat.shape[0])) ,self.SNR_mat.shape[0]), :, :,:].copy()
            
        self.H_mat  = self.H_mat[:,self.UA_set,:].copy()
        H = self.H_mat[0,:,:].copy()
        UA = np.zeros([self.UE]).astype(int)
        for i in range(self.UE):
            BS_ind = np.mod(i, self.BS)
            UE_ind = np.argmax(H[:,BS_ind])
            H[UE_ind,:] = -1.0
            UA[BS_ind * int(self.UE/self.BS) + int(i/self.BS)] = UE_ind
        self.H_mat = self.H_mat[:,UA,:].copy()
        self.H = self.H_mat[0, :,:].copy()
        
        self.QoS = np.random.choice(self.QoS_pool.shape[0],[self.UE,1])+0.0
        for i in range(self.QoS_pool.shape[0]):
            self.QoS[self.QoS == i] = self.QoS_pool[i]
        self.QoS[self.QoS==2.22] = (0.6 + (1.4-0.6)*np.random.uniform(size=  [np.sum(self.QoS==2.22)]) )
        self.QoS[self.QoS==4.44] = (2.0 + (6.0-2.0)*np.random.uniform(size=  [np.sum(self.QoS==4.44)]) )
        self.b_level = 100 * self.renew
        self.res_source =(self.renew_min+ (self.renew_max - self.renew_min)*np.random.uniform(size = [self.BS]))*self.renew
        self.state_PC = np.concatenate([(self.H*(self.b_level + self.grid_power - self.power_default)/260).reshape([1,-1])/2000.0, self.QoS.reshape([1,-1]), (np.maximum(self.b_level+self.grid_power - self.power_default,0.0).reshape([1,-1])/260.0), self.b_level.reshape([1,-1])/260, self.grid_power.reshape([1,-1])/260, (self.res_source).reshape([1,-1])/260,self.Backhaul.reshape([1,-1])/10.0], axis=1)
        
        
    def get_state_UA(self):
        self.P = np.clip( (self.renew * (0.01 + 0.69 * (self.action[0,0:self.BS]+1)/2)  + (1-self.renew) * (0.01 + 0.99 * (self.action[0,0:self.BS]+1)/2))*(self.b_level + self.grid_power -self.power_default)/self.delta/self.RB,
                        0, 1).reshape([1,-1])
        SNR = self.H * self.P
        SINR = SNR/ ( 1+ np.tile(np.sum(SNR,axis=1,keepdims=True),[1, self.BS]) - SNR)
        self.Rate = np.log2(1+SINR)*100*0.18 + 0.001 
        self.state_UA = np.concatenate([np.max(self.Rate,axis=0).reshape([1,-1]),-np.log(1+self.QoS/self.Rate).reshape([1,-1]),self.Backhaul.reshape([1,-1]), self.QoS.reshape([1,-1])],axis=1)        
        
    def get_X(self, Rate, QoS, Backhaul, mu, rho, is_print=False):
        mu = np.expand_dims(mu,axis=1)
        rho = np.expand_dims(rho,axis=1)
        Backhaul = np.expand_dims(Backhaul,axis=1)

        X = (np.expand_dims(np.argmax(Rate,axis=2),2) == np.arange(self.BS).reshape([1,1,self.BS]))+0.0
        lamb = np.max(Rate*X,axis=1,keepdims=True)
        count = 0
        while 1:
            lamb_old = lamb.copy()
            if X.shape[0] > 0:
                UE_order = np.random.permutation(self.UE)
            else:
                UE_order = np.argsort(np.min(np.maximum(QoS/Rate, QoS/Backhaul)[0,:,:],axis=1))
            for UE_ind in UE_order:
                X[:,UE_ind,:] = 0
                lamb = np.max(Rate*X,axis=1,keepdims=True)
                UE_opt = -(1+mu)*QoS * lamb/Rate - rho * QoS 
                ## Tie Break
                UE_select = np.argmax(UE_opt[:,UE_ind,:],axis=1)[0]
                BB=  -UE_opt[0,UE_ind,:].copy()
                indices = np.argsort(BB,axis=0)
                R_remain = 1-np.sum(np.sum(QoS/Rate*X,axis=1),axis=0)
                B_remain = Backhaul[0,0,:] - np.sum(np.sum(QoS*X,axis=1),axis=0)
                if R_remain[UE_select] < QoS[0,UE_ind,0]/Rate[0,UE_ind,UE_select] or B_remain[UE_select] < QoS[0,UE_ind,0]:
                    X[:,UE_ind,:] = 0.0
                    X[:,UE_ind,:] = (UE_select == np.arange(self.BS).reshape([1,self.BS]))+0.0
                    Y = self.get_Y(X[0,:,:],mu[0,:,:],rho[0,:,:])
                    reward_org = np.sum(self.Rate * X[0,:,:] * Y)/40 - np.sum(self.QoS > np.sum(self.Rate * X[0,:,:]*Y,axis=1,keepdims=True)+1e-7)/self.UE * 40
                    for B_ind in indices:
                        if abs(np.log(abs(BB[UE_select] / BB[B_ind])))<0.5:
                            X[:,UE_ind,:] = 0
                            X[:,UE_ind,:] = (B_ind == np.arange(self.BS).reshape([1,self.BS]))+0.0
                            Y=self.get_Y(X[0,:,:],mu[0,:,:],rho[0,:,:])
                            reward_new = np.sum(self.Rate * X[0,:,:] * Y)/40 - np.sum(self.QoS > np.sum(self.Rate * X[0,:,:]*Y,axis=1,keepdims=True)+1e-7)/self.UE * 40
                            if reward_new >reward_org:
                                UE_select = B_ind
                                break
                    
                X[:,UE_ind,:] = 0.0
                X[:,UE_ind,:] = (UE_select == np.arange(self.BS).reshape([1,self.BS]))+0.0
                lamb = np.max(Rate*X,axis=1,keepdims=True)
            if np.sum(abs(lamb_old-lamb)>1e-7) == 0:
                count = count+1
                if count > 1:
                    break
        Y = QoS / Rate * X #[Batch, UE, BS]
        Y_opt = Y.copy()
        Y_opt[Y_opt==0] = 9999999.9
        Y_s = np.sort(Y_opt,axis=1)
        QoS_tile = np.tile(QoS, [1,1,self.BS])
        ind = np.argsort(Y_opt,axis=1)
        QoS_s = np.take_along_axis(QoS_tile, ind, axis=1)
        fail_rate = 1-np.sum((np.cumsum(Y_s,axis=1) < 1) * (np.cumsum(QoS_s,axis=1)<Backhaul) )/self.UE
        return X.copy(), fail_rate
    def get_Y(self,X,mu,rho):
        Z = (np.argmax(self.Rate*X,axis=0).reshape([1,-1]) == np.arange(self.UE).reshape([-1,1]))* X+0.0
        
        Y = self.QoS/(self.Rate)*X
        for BS_ind in range(self.BS):
            while np.sum(Y[:,BS_ind]) > 1+1e-11 :
                ind = np.argmax(Y[:,BS_ind])
                Y[ind,BS_ind] = 0 #np.maximum(0, Y[ind,BS_ind] - (-1+ np.sum(Y[:,BS_ind])))
            while self.Backhaul[BS_ind] < np.sum(X[:,BS_ind]*self.Rate[:,BS_ind]*Y[:,BS_ind])-1e-11:
                ind = np.argmax((self.Rate * X * Y)[:,BS_ind])
                Y[ind,BS_ind] = 0# np.maximum(0, Y[ind,BS_ind]- (np.sum(X[:,BS_ind]*self.Rate[:,BS_ind]*Y[:,BS_ind])-self.Backhaul[BS_ind])/self.Rate[ind,BS_ind]   ) 
        Y = Y*(1-Z)
        Y = Y + Z* np.minimum( 1-np.sum(Y,axis=0,keepdims=True), np.tile(((self.Backhaul - np.sum(self.Rate*X*Y,axis=0))/np.sum(self.Rate*Z+0.00000001,axis=0)).reshape([1,self.BS]),[self.UE,1]))
        return Y

    def proceed_master(self, is_train = False):          
        mu = (self.action_UA[0,0: self.BS]).reshape([1,-1]).copy() # [1,BS]
        rho = (self.action_UA[0,self.BS: 2*self.BS]).reshape([1,-1]).copy() # [1,BS]
        if is_train:
            X = (np.argmax(self.Rate,axis=1).reshape([-1,1]) == np.arange(self.BS).reshape([1,-1]))
            fail_rate = 99
        else:
            X, fail_rate = self.get_X(self.Rate.reshape([1,self.UE,self.BS]).copy(), self.QoS.reshape([1,self.UE,1]).copy(), self.Backhaul.reshape([1,self.BS]).copy(), mu.copy(), rho.copy())
        X.shape = [self.UE, self.BS]
        Y = self.get_Y(X,mu,rho)
        ####################
        
        lamb = np.max(self.Rate*X,axis = 0,keepdims=True)
        fail_true = np.sum(self.QoS*X -  np.sum(self.Rate *X*Y, axis=1,keepdims=True) - 1e-5 > 0, axis = 0).reshape([1,self.BS])
        self.sum_rate = np.sum(self.Rate*X*Y)
        self.QoS_unsatisfactory = np.sum(self.QoS > np.sum(self.Rate *X*Y, axis=1,keepdims=True)+1e-7)/self.UE
        self.reward_PC_part1 = np.sum(self.Rate*X*Y,axis=0).reshape([1,self.BS])/40 
        self.reward_PC_part2 = -fail_true *40 / self.UE
        self.reward = np.concatenate([self.reward_PC_part1,self.reward_PC_part2],axis=1)
        self.QoS_over_amount = np.maximum(self.QoS.reshape([-1]) - np.sum(self.Rate*X*Y,axis=1).reshape([-1]),0.0)        
        
        
        self.Lag = np.sum((1+mu) * lamb *( 1- np.sum(self.QoS/self.Rate*X,axis=0,keepdims=True))) + np.sum( rho * (self.Backhaul.reshape([1,-1]) - np.sum(self.QoS*X,axis=0,keepdims=True)))
        
        ## New State
        Net_b_loss = np.maximum( self.power_default + self.RB * self.tx_w *  self.P * np.sum(X*Y,axis=0) * self.delta - self.grid_power,0) 
        self.X_save = X.copy()
        self.RES_save = self.res_source.copy()
        self.b_level = np.maximum(np.minimum( self.b_level - Net_b_loss + self.res_source,self.max_level ),0)
        self.res_source =  np.clip(3*np.random.normal(size=[self.BS]) +self.res_source, 37, 60) * self.renew
        
        self.H = self.H_mat[self.step+1, : ,:].copy()
        self.state_next = np.concatenate([(self.H*(self.b_level + self.grid_power - self.power_default)/260).reshape([1,-1])/2000.0, self.QoS.reshape([1,-1]), ((self.b_level + self.grid_power - self.power_default)/260).reshape([1,self.BS]), self.b_level.reshape([1,-1])/260, self.grid_power.reshape([1,-1])/260, (self.res_source).reshape([1,self.BS])/260,self.Backhaul.reshape([1,-1])/10.0], axis=1)
        
