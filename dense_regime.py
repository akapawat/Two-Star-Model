# -*- coding: utf-8 -*-
"""
Created on Fri May 22 18:17:51 2020

@author: pawat
"""
import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.optimize as scpo



 
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
        # print('not column fliped')
        # print(E_0,E_n)
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
        
        #flip type
        # if frames < int(int_frames*0.8) and np.random.rand() < 0.0001:
        #     E = big_flip(N,100,state,alpha,beta)
        # elif frames < int(int_frames*0.4) and np.random.rand() < 0.0001:
        #     E = big_flip(N,10,state,alpha,beta)
        if np.random.rand() < 0.0001:
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
        
        if frames % int(int_frames/200) ==0:
            k_0 = np.sum(state,axis=0)
            ydata.append(np.mean(k_0))
            # ydata.append(k_0)
            
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
    # state = np.zeros((N,N), dtype = 'int')
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

def p_eq(p,alpha,beta,N):
    return 0.5*(np.tanh(2*beta*p + alpha)+1)-p

def from_adj (N,alpha,beta):
    return np.loadtxt('adj matrix\\'+str(N)+'alpha'+str(alpha)+' finite_twostar'+str(beta)+'.adj',dtype = 'int')
  

def save_adj(N): 
    K = np.loadtxt('alpha_newton N=500.txt')
#    print(K)
    for chosen_i in (74,):
        alpha = K[chosen_i,0]
        beta = K[chosen_i,1]
        mean_deg = K[chosen_i,2]
        
        
        print('alpha : %.4f, beta : %.4f, <k> = %.4f, N = %d'%(alpha,beta,mean_deg,N))
        state1 = ini_fixed(N,int(mean_deg*N/2))
#        state1 = np.loadtxt('saved matrix\\500alpha-3.023359 finite_twostar0.0154.adj')
        
        k_0 = np.sum(state1,axis=0)
        print(np.mean(np.power(k_0,2)))
       
        flip(1e8,state1,N,alpha,beta)
        
        k_0 = np.sum(state1,axis=0)
        print(np.mean(np.power(k_0,2)))
        plt.show()
        np.savetxt('saved matrix\\'+str(N)+'alpha'+str(alpha)+' finite_twostar'+str(beta)+'.adj',state1)

def main (N,steps):
     
    beta = np.append(np.arange(0,0.95,0.1),np.append(np.arange(0.95,1.05,0.02),np.arange(1.1,2,0.1)))
    alpha = -beta
    
    FileName = ' test non-sparse N= %d.txt'%(N,)
    
    file = open(FileName,'w')
    
    xdata = []
    ydata1 = []
    ydata2 = []
    ydatanumer = []
    
    # print(ydata2)
    
    print('N =',N)
    
#    for i in (11,):
    for i in range(len(beta)):
        
        datum1 = []
        datum2 = []
        
        
# =============================================================================
#         # Initialize adj
# =============================================================================
        # state1 = np.zeros((N,N))
        # state1 = ini_random(N)
#       state1 = from_adj(beta[i])
        state1 = np.float32(np.delete(np.delete(np.loadtxt('saved matrix\\'+str(N)+'alpha'+str(alpha[i])+'twostar'+str(beta[i])+'.csv',dtype = str,delimiter =';'),0,0),0,1))
        
        
        # print(ydatanumer)
        # if beta[i] > 0.8 and beta[i] < 1.2:
        flip(steps,state1,N,alpha[i],beta[i]/(N-1))
        # else :
        #     flip(steps*0.25,state1,N,alpha[i],beta[i]/(N-1))

#        np.savetxt('adj matrix\\'+str(N)+'alpha'+str(alpha)+' finite_twostar'+str(beta[i])+'.adj',state1)

        for itet in (2e2,)*100:
            flip(itet,state1,N,alpha[i],beta[i]/(N-1),False)
            deg = np.sum(state1,axis = 0)
            datum1.append(np.mean(deg))
            datum2.append(np.mean(np.power(deg,2))-np.power(np.mean(deg),2))
            
        data1 = np.mean(datum1)    
        data2 = np.mean(datum2)
        error1 = np.std(datum1)
        error2 = np.std(datum2)
        print (beta[i], data1,data2)
        xdata = np.append(xdata,beta[i])
        ydata1 = np.append(ydata1,data1)
        ydata2 = np.append(ydata2,data2)
        file.write(str(beta[i])+'\t'+str(data1)+'\t'+str(error1)+'\t'+str(data2)+'\t'+str(error2)+'\n')
        
        save_state = np.concatenate((np.transpose([np.arange(N,dtype = 'float')]),state1),axis=1)
        save_state = np.concatenate(([np.concatenate(([''],np.arange(N,dtype = 'float')))],save_state),axis=0)
        np.savetxt('saved matrix\\'+str(N)+'alpha'+str(alpha[i])+'twostar'+str(beta[i])+'.csv',save_state,fmt='%s',delimiter =';')
    
    file.close()

def plot_data():
    A = np.loadtxt('non-sparse w er N= 300.txt')
    beta = A[:,0]
    beta_num = np.delete(np.arange(0,2,0.01),100)
    alpha = -beta
    alpha_num = -beta_num
    ydata1 = A[:,1]
    ydata2 = A[:,3]
    ydatanumer = []
    ydatanumer1 = []
    ydatanumer2 = []
    N = 300
    y_error1 = A[:,2]
    y_error2 = A[:,4]
    
    fig = plt.figure(figsize = (8,6))
    ax = plt.axes()

    x= np.arange(0,2,0.01)
    plt.plot(x,[p_eq(i,-1.5,1.5,N) for i in x])
    plt.show()
    
    for i in range(len(beta_num)):
        p = scpo.newton(p_eq,0.5,args=(alpha_num[i],beta_num[i],N))
        ydatanumer1.append((N-1)*p+2*beta_num[i]*p*(1-p)*(1-2*p)/(1-4*beta_num[i]*p*(1-p))/(1-2*beta_num[i]*p*(1-p)))
        # p = scpo.newton(p_eq,0,args=(alpha_num[i],beta_num[i],N))
        # ydatanumer2.append((N-1)*p+2*beta_num[i]*p*(1-p)*(1-2*p)/(1-4*beta_num[i]*p*(1-p))/(1-2*beta_num[i]*p*(1-p)))
        #ydatanumer.append(p*(1-p)/(1-2*beta_num[i]*p*(1-p)))
        
    
    plt.xlim(0,2)
#        ax.set_ylim(-0.7,1)
    # plt.errorbar(beta,(ydata2/(N-1)),yerr = y_error2/(N-1),fmt = 'o',elinewidth = 1,label = 'MCMC')
    # plt.errorbar(beta,(ydata1/(N-1)),yerr = y_error1/(N-1),fmt = 'o',elinewidth=1, label = 'MCMC')
    plt.plot(beta_num,np.array(ydatanumer1)/(N-1),'orange', label = 'MF')
    # plt.plot(beta_num,np.array(ydatanumer2)/(N-1),'orange')
    
    plt.xlabel('J',fontsize = 14)
    plt.ylabel('$<k>/(n-1)$',fontsize = 14)
    plt.legend()
    # plt.savefig("non_sparse_test")
    plt.show()
    
    
# main(300,200)
plot_data()


#save_adj(500)
