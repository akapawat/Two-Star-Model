# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 22:09:21 2019

@author: pawat
"""

import numpy as np
import matplotlib.pyplot as plt
import random


 
def cal_energy (state,alpha,beta):
    k_0 = np.sum(state,axis=0)
    k2_0 = np.power(k_0,2)
    return -beta*np.sum(k2_0) -alpha*np.sum(k_0)

def normal_flip(N,state,alpha,beta):
    E_0 = cal_energy(state,alpha,beta)
    
    i = np.random.randint(1,N)
    j = np.random.randint(0,i)

    
    state[j,i] = 1-state[j,i]
    state[i,j] = state[j,i]
    
    E_n = cal_energy(state,alpha,beta)
    
    if not(np.random.random() < min(1,np.exp(E_0-E_n))): 
        state[j,i] = 1-state[j,i]
        state[i,j] = state[j,i]
        return E_0
    
    return E_n

def column_flip(N,state,alpha,beta):
    E_0 = cal_energy(state,alpha,beta)
    
    i = np.random.randint(0,N)

    state[:,i] = 1-state[:,i]
    state[i,:] = 1-state[i,:]
    
    E_n = cal_energy(state,alpha,beta)
    
    if not(np.random.random() < min(1,np.exp(E_0-E_n))): 
        state[:,i] = 1-state[:,i]
        state[i,:] = 1-state[i,:]

        return E_0
    
    return E_n

def big_flip(N,n,state,alpha,beta):
    E_0 = cal_energy(state,alpha,beta)
    # print('big flip called')
    state0 = state.copy()
    
    set_ij = np.sort(np.random.randint(0,N,size = (n,2)))
    
    for thing in set_ij: 
        i = thing[1]
        j = thing[0]
        if i!=j:
            state[j,i] = 1-state[j,i]
            state[i,j] = state[j,i]
    
    E_n = cal_energy(state,alpha,beta)
    
    if not(np.random.random() < min(1,np.exp(E_0-E_n))): 
        state = state0.copy()
        
        # print('no flipped')
        # print(E_0)
        # print(cal_energy(state, alpha, beta))
        return E_0
        
    return E_n

def edge_swap(N,state,alpha,beta):
    E_0 = cal_energy(state,alpha,beta)
    
    set_i = np.random.randint(1,high = N,size=2)
    i1 = int(set_i[0])
    j1 = np.random.randint(0,high = i1)
    i2 = int(set_i[1])
    j2 = np.random.randint(0,high = i2)
    
    temp = state[j1,i1]
    state[j1,i1] = state[j2,i2]
    state[j2,i2] = temp
    state[i1,j1] = state[j1,i1]
    state[i2,j2] = state[j2,i2]
    
    E_n = cal_energy(state,alpha,beta)
#    print(E_n-E_0)
    
    if not(np.random.random() < min(1,np.exp(E_0-E_n))): 
        temp = state[j1,i1]
        state[j1,i1] = state[j2,i2]
        state[j2,i2] = temp
        state[i1,j1] = state[j1,i1]
        state[i2,j2] = state[j2,i2]
        return E_0
    return E_n

def hinge_flip(N,state,alpha,beta):
    E_0 = cal_energy(state,alpha,beta)
    '''
    set_i = np.random.randint(1,high = N,size=2)
    i1 = int(set_i[0])
    j1 = np.random.randint(0,high = i1)
    i2 = int(set_i[1])
    j2 = np.random.randint(0,high = i2)
    
    temp = state[j1,i1]
    state[j1,i1] = state[j2,i2]
    state[j2,i2] = temp
    state[i1,j1] = state[j1,i1]
    state[i2,j2] = state[j2,i2]
    '''
    
    i1 = np.random.randint(1,high=N)
    j1 = np.random.randint(0,high = i1)
    j2 = np.random.randint(0,high = i1)
    
    temp = state[j1,i1]
    state[j1,i1] = state[j2,i1]
    state[j2,i1] = temp
    state[i1,j1] = state[j1,i1]
    state[i1,j2] = state[j2,i1]
    
    E_n = cal_energy(state,alpha,beta)
#    print(E_n-E_0)
    
    if not(np.random.random() < min(1,np.exp(E_0-E_n))): 
        temp = state[j1,i1]
        state[j1,i1] = state[j2,i1]
        state[j2,i1] = temp
        state[i1,j1] = state[j1,i1]
        state[i1,j2] = state[j2,i1]
        
        return E_0
    #undirected graph
    return E_n

        
        
def flip(frames, state,N,alpha, beta, plot_t = True): #state2 has higher Temp
    ydata = []
    c_E = []
    int_frames = frames
    Tw = 20000
    
    while(frames > 0):
        
# =============================================================================
#         #flip type
# =============================================================================
        # if frames < int(int_frames*0.8) and np.random.rand() < 0.0001:
        #     E = big_flip(N,100,state,alpha,beta)
        # elif frames < int(int_frames*0.4) and np.random.rand() < 0.0001:
        #     E = big_flip(N,10,state,alpha,beta)
        if np.random.rand() < 0.001:
            E = column_flip(N,state,alpha,beta)
        else:
            E = normal_flip(N,state,alpha,beta)
            
            
        # Terminate by checking E
#         c_E.append(E)
# #        print(c_E)
#         if frames % (Tw*2) == 0 and frames!=int_frames:
# #            print('check at', frames)
#             if np.abs(np.sum(c_E[:Tw])/Tw - np.sum(c_E[Tw:])/Tw) <= 0.001:
#                 state = 1-state
#                 print(np.sum(c_E[:Tw])/np.sum(c_E[Tw:]))
#                 print('end at',frames)
#                 break
#             c_E.clear()
        
        if plot_t:
            if frames % int(int_frames/200) ==0:
                k_0 = np.sum(state,axis=0)
                ydata.append(np.mean(np.power(k_0,2)))
                # ydata.append(E)
    #        if frames % int(int_frames/10) == 0:
    #            print(frames)
    #        print(ydata)
        frames -= 1
        
        
        '''J *= factor'''
#    print(ydata)
    if plot_t:
        # print('plot')
        plt.plot(ydata)
        plt.show()
   # deg = np.sum(state1,axis = 0)
    #print('connectence is ', np.mean(deg)/(N-1))

def p_glauber(twostar1,twostar2,N,beta1,beta2):
    
    return 1/(np.exp(2*beta1+beta2/N*(twostar2-twostar2))+1)

def F_glauber(state,i,j,rand,N,beta1,beta2):
    k1 = np.sum(state, axis=0)
    twostar1 = np.sum(np.power(k1,2)) - np.sum(k1)
    
    state[j,i] = 1-state[j,i]
    state[i,j] = state[j,i]
    
    k2 = np.sum(state, axis=0)
    twostar2 = np.sum(np.power(k2,2)) - np.sum(k2)
    
    if not(rand < p_glauber(twostar1,twostar2,N,beta1,beta2)): 
        state[j,i] = 1-state[j,i]
        state[i,j] = state[j,i]

def glauber(state,N,alpha, beta):
    n = 0
    beta2 = N*beta/2
    beta1 = (alpha+beta)/2
    
    lower = np.zeros((N,N))
    upper = np.ones((N,N))
    for i in range(N):
        upper[i,i] = 0
    
    while(upper != lower):
        n+=1
        
        i = np.random.randint(1,N)
        j = np.random.randint(0,i)
        rand = np.random.random()
        
        F_glauber(upper,i,j,rand,N,beta1,beta2)
        F_glauber(lower,i,j,rand,N,beta1,beta2)
        
        
        
        
        

def ini_random(N):
    state = np.random.randint(2,size = (N,N))
    
    for i in range (N):
        for j in range(N):
            if (i==j):
                state[j,i] = 0
            elif (i>j):
                state[i,j] = state[j,i]

    return state

def ini_fixed(N,K):
    # print (np.random.choice(N,size=K,replace=False))
    state = np.zeros((N,N), dtype = 'int')
    # set_ij = [[j,i] for i in range(1,N) for j in range(i)]
    # for ij in random.sample(set_ij,k = K):
    #     state[ij[0],ij[1]] = 1

#    for i in range(N):
#        for j in np.random.choice(N,size=K,replace=False):
#            state[j,i] = 1

    
    for i in range (N):
        for j in range(N):
            if (i==j):
                state[j,i] = 0
            elif (i>j):
                state[i,j] = state[j,i]
#        print(state)
#        print(K)
#        print(np.mean(np.sum(state,axis = 0)))
    return state

def from_adj (N,alpha,beta):
    return np.loadtxt('adj matrix\\'+str(N)+'alpha'+str(alpha)+' finite_twostar'+str(beta)+'.adj',dtype = 'int')
  

def save_adj(N,steps): 


    alpha = -0.5-0.5*np.log(N)
    beta = 0.0001
    
    
    print('alpha : %.4f, beta : %.4f, N = %d'%(alpha,beta,N))
    # state1 = ini_random(N)
    state1 = np.float32(np.delete(np.delete(np.loadtxt('saved matrix\\'+str(N)+'alpha'+str(alpha)+'twostar'+str(beta)+'.csv',dtype = str,delimiter =';'),0,0),0,1))
    
    k_0 = np.sum(state1,axis=0)
    print(np.mean(np.power(k_0,2)))
   
    flip(steps,state1,N,alpha,beta)
    
    k_0 = np.sum(state1,axis=0)
    print(np.mean(np.power(k_0,2)))
    
    save_state = np.concatenate((np.transpose([np.arange(N,dtype = 'float')]),state1),axis=1)
    save_state = np.concatenate(([np.concatenate(([''],np.arange(N,dtype = 'float')))],save_state),axis=0)
    np.savetxt('saved matrix\\'+str(N)+'alpha'+str(alpha)+'twostar'+str(beta)+'.csv',save_state,fmt='%s',delimiter =';')

def main (N,steps):
     
    alpha = 4-0.5*np.log(N)
    beta = np.arange(-1,0,0.05)
    
    fig = plt.figure()
    ax = plt.axes()
    ax.set_title ("Random Exp  Graph")
    
    
    FileName = 'alpha = %.2f, N= %d.txt'%(alpha,N)
    
    file = open(FileName,'w')
    
    xdata = []
    ydata1 = []
    ydata2 = []

    
    print('N =',N,'alpha =',alpha)

    for i in range(len(beta)):
        
        datum1 = []
        datum2 = []
        
        
# =============================================================================
#         # Initialize adj matrix
# =============================================================================
        # state1 = np.zeros((N,N))
        # state1 = ini_random(N)
        # state1 = from_adj(beta[i])
        state1 = np.float32(np.delete(np.delete(np.loadtxt('saved matrix\\'+str(N)+'alpha'+str(alpha[i])+'twostar'+str(beta[i])+'.csv',dtype = str,delimiter =';'),0,0),0,1))
        
        
        flip(steps,state1,N,alpha,beta[i])


        for itet in (1e2,)*10:
            flip(itet,state1,N,alpha,beta[i],False)
            deg = np.sum(state1,axis = 0)
            datum1.append(np.mean(deg))
            datum2.append(np.mean(np.power(deg,2))-np.power(np.mean(deg),2))
            
        data1 = np.mean(datum1)    
        data2 = np.mean(datum2)
        print (beta[i], data1, data2)
        xdata = np.append(xdata,beta[i])
        ydata1 = np.append(ydata1,data1)
        ydata2 = np.append(ydata2,data2)
        file.write(str(beta[i])+'\t'+str(data1)+'\t'+str(data2)+'\n')
        

        save_state = np.concatenate((np.transpose([np.arange(N,dtype = 'float')]),state1),axis=1)
        save_state = np.concatenate(([np.concatenate(([''],np.arange(N,dtype = 'float')))],save_state),axis=0)
        np.savetxt('saved matrix\\'+str(N)+'alpha'+str(alpha)+'twostar'+str(beta[i])+'.csv',save_state,fmt='%s',delimiter =';')
    
#        ax.set_xlim(-1,1)
#        ax.set_ylim(-0.7,1)
    # plt.plot(xdata,(ydata1/N),'bo',label = 'conectivity')
    # plt.plot(xdata,ydata2,'ro',label = 'var')
    #resahpe matrix
    file.close()
    plt.show()


# main(300,1e4)

save_adj(1000,5e6)
