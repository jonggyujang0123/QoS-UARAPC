import tensorflow as tf
import numpy as np
import random 



class GaussianNoise():
    def __init__(self,action_dimension,epsilon_init = 0.7, epsilon_end = 0.3,mu=0, theta =0.15, sigma = 0.25):
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.epsilon_decay = 0.9995
        self.epsilon = epsilon_init
        self.epsilon_end = epsilon_end
        self.decay = (epsilon_init-epsilon_end)/10000.
    def reset(self):
        self.epsilon = np.maximum(self.epsilon - self.decay, self.epsilon_end)
        
    def noise(self,step):
        self.is_noise = (np.random.uniform() <self.epsilon)
        noise = np.random.normal(size= [1,self.action_dimension])* self.sigma  * self.is_noise 
        return noise
        
class Actor_USL():
    def __init__(self, action_size, scope = 'DDPG_Actor'):
        self.output_size = action_size
        self.scope = scope
        
    def forward(self,state):
        with tf.variable_scope(self.scope, reuse = tf.AUTO_REUSE):
            self.state = state
            self.fcn1 = tf.contrib.layers.fully_connected(self.state, 2048, activation_fn = tf.nn.leaky_relu)
            self.fcn2= tf.contrib.layers.fully_connected(self.fcn1, 2048, activation_fn = tf.nn.leaky_relu)
            self.fcn3= tf.contrib.layers.fully_connected(self.fcn2, 2048, activation_fn =tf.nn.leaky_relu)
            self.fcn4= tf.contrib.layers.fully_connected(self.fcn3, 2048, activation_fn =tf.nn.leaky_relu)
            self.fcn5= tf.contrib.layers.fully_connected(self.fcn4, 2048, activation_fn =tf.nn.leaky_relu)
            self.fcn6= tf.contrib.layers.fully_connected(self.fcn5, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn7= tf.contrib.layers.fully_connected(self.fcn6, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn8= tf.contrib.layers.fully_connected(self.fcn7, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn9= tf.contrib.layers.fully_connected(self.fcn8, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn10= tf.contrib.layers.fully_connected(self.fcn9, 512, activation_fn =tf.nn.leaky_relu)
            self.fcn11= tf.contrib.layers.fully_connected(self.fcn10, 256, activation_fn =tf.nn.leaky_relu)
            self.fcn12= tf.contrib.layers.fully_connected(self.fcn11, 128, activation_fn =tf.nn.leaky_relu)
            self.action = 1+tf.nn.elu(tf.contrib.layers.fully_connected(self.fcn12,self.output_size, activation_fn = None))
        return self.action

class Actor():
    def __init__(self, action_size, thr, scope = 'DDPG_Actor', is_tanh = True):
        self.output_size = action_size
        self.scope = scope
        self.is_tanh = is_tanh
        self.thr = thr
        
    def forward(self,state):
        with tf.variable_scope(self.scope, reuse = tf.AUTO_REUSE):
            ## Actor
            state_part1 = state[:,0:self.thr]
            state_part2 = state[:, self.thr::]
            state_part1_ = tf.contrib.layers.fully_connected(state_part1, 512, activation_fn = tf.nn.leaky_relu)
            state_part2_ = tf.contrib.layers.fully_connected(state_part2, 512, activation_fn = tf.nn.leaky_relu)
            state_post   = tf.concat([state_part1_, state_part2_],axis=1)
            self.fcn1 = tf.contrib.layers.fully_connected(state_post, 2048, activation_fn = tf.nn.leaky_relu)
            self.fcn2= tf.contrib.layers.fully_connected(self.fcn1, 2048, activation_fn = tf.nn.leaky_relu)
            self.fcn3= tf.contrib.layers.fully_connected(self.fcn2, 2048, activation_fn =tf.nn.leaky_relu)
            self.fcn4= tf.contrib.layers.fully_connected(self.fcn3, 2048, activation_fn =tf.nn.leaky_relu)
            self.fcn5= tf.contrib.layers.fully_connected(self.fcn4, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn6= tf.contrib.layers.fully_connected(self.fcn5, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn7= tf.contrib.layers.fully_connected(self.fcn6, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn8= tf.contrib.layers.fully_connected(self.fcn7, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn9= tf.contrib.layers.fully_connected(self.fcn8, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn10= tf.contrib.layers.fully_connected(self.fcn9, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn11= tf.contrib.layers.fully_connected(self.fcn10, 512, activation_fn =tf.nn.leaky_relu)
            self.fcn12= tf.contrib.layers.fully_connected(self.fcn11, 512, activation_fn =tf.nn.leaky_relu)
            self.action = tf.tanh(tf.contrib.layers.fully_connected(self.fcn12,self.output_size, activation_fn = None))
        return self.action
   
class Critic():
    def __init__(self, reward_size, BS, UE, scope = 'DDPG_Critic'):
        self.scope = scope
        self.reward_size = reward_size
        self.BS = BS
        self.UE = UE
        self.renew = (np.arange(self.BS) != self.BS-1).astype(int) #np.array([1,1,1,0])
    def state_action_to_PCstate(self, state, action):
        P = tf.reshape((self.renew * (0.01 + 0.69 * (action+1)/2)  + (1-self.renew) * (0.01 + 0.99 * (action+1)/2)), [-1, 1, self.BS])
        SNR_p = 2000*tf.reshape(state[:,0:self.UE*self.BS],[-1, self.UE,self.BS]) * P
        SINR = SNR_p/ ( 1+ tf.reduce_sum(SNR_p,axis=2,keepdims=True)- SNR_p)
        Rate = tf.log(1+SINR)/tf.log(2.0)*18 + 0.001         
        QoS = tf.reshape(state[:,self.UE*self.BS:self.UE*self.BS + self.UE ], [-1, self.UE, 1])
        Avail_energy = state[:,self.UE*self.BS + self.UE  : self.UE*self.BS + self.UE  + self.BS] 
        grid_power=  state[:, self.UE*self.BS + self.UE  + 2 * self.BS : self.UE*self.BS + self.UE  + 3 *self.BS]
        RES= state[:, self.UE*self.BS + self.UE  + 3 * self.BS : self.UE*self.BS + self.UE  + 4 *self.BS]
        Backhaul = state[:, self.UE*self.BS + self.UE  + 4 * self.BS : self.UE*self.BS + self.UE  + 5 *self.BS]
        
        
        state_1 = tf.reshape(-tf.log(QoS/Rate), [-1, self.BS * self.UE]) # QoS-Rate Ratio [-1, BS*UE]
        state_2 = tf.reshape( -tf.log(QoS / 10 /tf.reshape(Backhaul,[-1, 1, self.BS])), [-1, self.BS * self.UE]) # QoS-Bh Ratio [-1, BS * UE]
        state_3 = -tf.log(self.renew * Avail_energy * tf.reshape(1-P, [-1,self.BS]) +RES + grid_power) # Remaining energy [-1, BS]
        state_4 = tf.reduce_max(Rate, axis=1)/100.0 # Max_Rate [-1,BS]
        state_5 = RES + 0.0 # RES [-1, BS]
        
        return tf.concat([state_1, state_2],axis=1), tf.concat([state_3, state_4, state_5], axis=1)
        
    def forward(self,state, action):
        with tf.variable_scope(self.scope, reuse = tf.AUTO_REUSE): 
            state_part1, state_part2 = self.state_action_to_PCstate(state, action)
            state_part1_ = tf.contrib.layers.fully_connected(state_part1, 512, activation_fn = tf.nn.leaky_relu)
            state_part2_ = tf.contrib.layers.fully_connected(state_part2, 512, activation_fn = tf.nn.leaky_relu)
            state_post   = tf.concat([state_part1_, state_part2_],axis=1)
            self.fcn1 = tf.contrib.layers.fully_connected(state_post, 2048, activation_fn =tf.nn.leaky_relu)
            self.fcn2= tf.contrib.layers.fully_connected(self.fcn1, 2048, activation_fn =tf.nn.leaky_relu)
            self.fcn3= tf.contrib.layers.fully_connected(self.fcn2, 2048, activation_fn =tf.nn.leaky_relu)
            self.fcn4= tf.contrib.layers.fully_connected(self.fcn3, 2048, activation_fn =tf.nn.leaky_relu)
            self.fcn5= tf.contrib.layers.fully_connected(self.fcn4, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn6= tf.contrib.layers.fully_connected(self.fcn5, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn7= tf.contrib.layers.fully_connected(self.fcn6, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn8= tf.contrib.layers.fully_connected(self.fcn7, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn9= tf.contrib.layers.fully_connected(self.fcn8, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn10= tf.contrib.layers.fully_connected(self.fcn9, 1024, activation_fn =tf.nn.leaky_relu)
            self.fcn11= tf.contrib.layers.fully_connected(self.fcn10, 512, activation_fn =tf.nn.leaky_relu)
            self.fcn12= tf.contrib.layers.fully_connected(self.fcn11, 512, activation_fn =tf.nn.leaky_relu)
            self.Qval = tf.contrib.layers.fully_connected(self.fcn12,self.reward_size,activation_fn = None)
        return self.Qval
        
        
class DDPG():
    def __init__(self, scope, sess, BS, UE, Actor , Critic,Actor_target , Critic_target, OUNoise, replay_buffer, state_size, action_size,reward_size, gamma, lr_actor, lr_critic, batch_size,tau,is_tanh):
        self.sess = sess
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.scope = scope
        self.is_tanh = is_tanh
        self.gamma = gamma
        self.Actor = Actor
        self.Critic = Critic
        self.Actor_target = Actor_target
        self.Critic_target = Critic_target
        self.noise = OUNoise
        self.replay_buffer = replay_buffer
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.reward_size = reward_size
        self.state = np.zeros([1,state_size])
        self.action = np.zeros([1, action_size])
        self.state_next = np.zeros([1,state_size])
        self.reward = np.zeros([1,self.reward_size])
        self.state_ph = tf.placeholder(shape = [None,state_size], dtype = tf.float32)
        self.action_ph = tf.placeholder(shape = [None,action_size], dtype = tf.float32)
        self.state_ph_next = tf.placeholder(shape = [None,state_size], dtype= tf.float32)
        self.reward_ph = tf.placeholder(shape = [None,self.reward_size], dtype = tf.float32)
        self.BS = BS
        self.UE = UE
        # Network models + Actor netowrk update
        self.action_tf = self.Actor.forward(self.state_ph)
        self.qval = self.Critic.forward(self.state_ph, self.action_tf)
        self.gradient_action = tf.reshape(tf.gradients(tf.reduce_sum(self.qval),self.action_tf),[-1,self.action_size]) 
        self.target_action = tf.clip_by_value(tf.stop_gradient(self.action_tf + 0.03*self.gradient_action),-0.99,0.99)
            
        self.loss_weight = tf.placeholder(shape= [None,1], dtype = tf.float32)
        self.policy_loss = tf.reduce_mean(self.loss_weight*tf.reduce_mean((self.target_action-self.action_tf)**2,axis=1,keepdims=True))
        self.train_policy = tf.train.AdamOptimizer(learning_rate = self.lr_actor).minimize(self.policy_loss)
        
        ## Critic netowrk update
        
        self.action_next_tf = self.Actor_target.forward(self.state_ph_next)
    
        self.target_qval = tf.stop_gradient(self.Critic_target.forward(self.state_ph_next, self.action_next_tf))
        self.target_critic = self.reward_ph + self.gamma * self.target_qval
        self.loss_critic = tf.reduce_mean(self.loss_weight * tf.reduce_mean((self.target_critic - self.Critic.forward(self.state_ph, self.action_ph))**2,axis=1,keepdims=True))
        self.TD_error = tf.sqrt(tf.reduce_sum(tf.abs(self.target_critic - self.Critic.forward(self.state_ph, self.action_ph))**2,axis=1,keepdims=True))
        self.loss_critic_wo_noise = tf.reduce_mean(tf.reduce_mean((self.target_critic - self.Critic.forward(self.state_ph, self.action_ph))**2,axis=1,keepdims=True))
        self.train_critic = tf.train.AdamOptimizer(learning_rate = self.lr_critic).minimize(self.loss_critic)
        self.Actor_noiseless_tf = self.Actor_target.forward(self.state_ph)
        tfVars = tf.trainable_variables(scope = self.scope )
        tau = self.tau
        total_vars = len(tfVars)
        self.op_holder =[]
        for index, var in enumerate(tfVars[0:int(total_vars/2)]):
            self.op_holder.append(tfVars[index+int(total_vars/2)].assign((var.value()*tau)+((1-tau)*tfVars[index+int(total_vars/2)].value())))
    
    
    def add_exp(self, state, state_next, action, reward):
        self.replay_buffer.add(state, state_next, action, reward)
    
    def forward_test_action(self,state):
        return self.sess.run(self.Actor_noiseless_tf, feed_dict = {self.state_ph : state})

    def forward_noiseless_action(self,state):
        return self.sess.run(self.action_tf, feed_dict = {self.state_ph : state})
        
    def forward_noise_action(self,state, step):
        if self.is_tanh == True:
            output = np.clip(self.sess.run(self.action_tf, feed_dict = {self.state_ph : state}) + self.noise.noise(step), -1., 1.)
        else:
            output = np.clip(self.sess.run(self.action_tf, feed_dict = {self.state_ph : state}) + self.noise.noise(), 0.00, 1000.)
        return output
        
    def forward_loss(self,s,s_1,a,r):
        return self.sess.run(self.loss_critic_wo_noise, feed_dict = {self.state_ph : s, self.action_ph: a, self.state_ph_next: s_1, self.reward_ph : r})
        
        
class PER():
    def __init__(self, buffer_size = 10000, alpha = 0.4, epsilon_per = 0.001, beta = 0.7):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon_per
        self.buffer_size = buffer_size
        self.buffer = []
        self.prob_bean = np.zeros([0])
        self.alpha_decay = (self.alpha- 0.0)/15000
        self.beta_increasing = (1.0-self.beta)/15000
        
    def add(self, s,s_1,a,r, ):
        self.buffer.append((s,s_1,a,r))
        if self.prob_bean.shape[0] == 0:
            self.prob_bean = np.concatenate([self.prob_bean,[self.epsilon]],axis=0)
        else:
            self.prob_bean = np.concatenate([self.prob_bean,[max(self.prob_bean)]],axis=0)
            
        
        if len(self.buffer) == self.buffer_size +1 :
            self.prob_bean = self.prob_bean[1:self.buffer_size+1]
            del self.buffer[0]
        
            
    def sample(self, batch_size):
        self.alpha = np.maximum(self.alpha-self.alpha_decay, 0.0)
        self.beta = np.minimum(self.beta +self.beta_increasing, 1.0)
        batch =list()
        idx = np.random.choice(range(len(self.buffer)),size = batch_size, replace = False, p = self.prob_bean**self.alpha/sum(self.prob_bean**self.alpha))
        for i in range(batch_size):
            batch.append(self.buffer[idx[i]])
        
        s, s_1, a, r = zip(*batch)
        s = np.concatenate(s)
        s_1 = np.concatenate(s_1)
        a = np.concatenate(a)
        r = np.concatenate(r)
        loss_weight = (1/self.prob_bean[idx]**self.alpha * sum(self.prob_bean**self.alpha)/ len(self.buffer) )**self.beta
        loss_weight = loss_weight/max(loss_weight)
        return s, s_1, a, r, loss_weight, idx
    
    def update_weight(self, idx, TD_error):
        self.prob_bean[idx] = (TD_error.reshape([-1]) + self.epsilon)
        

class USL():
    def __init__(self, scope, sess, BS, UE, Actor , replay_buffer, state_size, action_size, lr_actor, batch_size, alpha_init):
        self.sess = sess
        self.batch_size = batch_size
        self.lr_actor = lr_actor
        self.scope = scope
        self.Actor = Actor
        self.replay_buffer = replay_buffer
        self.state_size = state_size
        self.action_size = action_size
        self.state = np.zeros([1,state_size])
        self.action = np.zeros([1, action_size])
        self.state_ph = tf.placeholder(shape = [None,state_size], dtype = tf.float32)
        self.BS = BS
        self.UE = UE
        self.radius = 4 * self.BS**0.5
        self.mu_ind = np.concatenate([np.ones([1,self.BS]), np.zeros([1,self.BS])],axis=1)
        # Network models + Actor netowrk update
        
        self.action_tf = self.Actor.forward(self.state_ph)
        self.target_action= tf.placeholder(shape = [None,action_size], dtype = tf.float32)
        self.loss = tf.reduce_mean((self.target_action - self.action_tf)**2)
        self.train_weights = tf.train.AdamOptimizer(learning_rate = self.lr_actor).minimize(self.loss)
        
        self.alpha = alpha_init
        
    def add_exp(self, state, Rate, QoS, Backhaul):
        self.replay_buffer.add(state, Rate, QoS, Backhaul)
    
    def forward_action(self,state):
        return self.sess.run(self.action_tf, feed_dict = {self.state_ph : state})

        
class USL_replay():
    def __init__(self, buffer_size = 10000):
        self.buffer_size = buffer_size
        self.buffer = []
        
    def add(self, State, Rate, QoS, Backhaul):
        Rate = np.expand_dims(Rate,0)
        QoS = np.expand_dims(QoS,0)
        Backhaul = np.expand_dims(Backhaul,0)
        self.buffer.append((State, Rate,QoS,Backhaul))    
        if len(self.buffer) == self.buffer_size +1 :
            del self.buffer[0]
        
            
    def sample(self, batch_size):
        batch =list()
        idx = np.random.choice(range(len(self.buffer)),size = batch_size, replace = False)
        for i in range(batch_size):
            batch.append(self.buffer[idx[i]])
        State, Rate, QoS, Backhaul = zip(*batch)
        State = np.concatenate(State)
        Rate = np.concatenate(Rate)
        QoS = np.concatenate(QoS)
        Backhaul = np.concatenate(Backhaul)
        return State, Rate, QoS, Backhaul
        
        
        
