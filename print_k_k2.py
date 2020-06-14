# -*- coding: utf-8 -*-
"""
Created on Tue May 12 18:28:10 2020

@author: pawat
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as scpo

#summation term in (55) (56)
def sum_k (order,kmean,trunc,alpha,beta,N):
    #c = 1
    summation = 0
    for k in range(trunc+1):
        prod = 1
        for n in range(k):
            n+=1
            prod *= np.exp(alpha+beta*k)/n
#        print('prod',prod)
        temp_k = np.power(k,order)*np.power(kmean,k/2)*np.exp((-kmean-1)/2)*prod
        # print(k,temp_k)
        summation += temp_k
#        if temp_k > 1e11:
##            print(k)
#            break
#    print(summation)
    # print(k)
    return summation

#eq(55)
def finite_k(alpha,beta,kmean,trunc,N):
    return sum_k(1,kmean,trunc,alpha,beta,N)/sum_k(0,kmean,trunc,alpha,beta,N)

#eq(56)
def finite_k2(alpha,beta,kmean,trunc,N):
    return sum_k(2,kmean,trunc,alpha,beta,N)/sum_k(0,kmean,trunc,alpha,beta,N)

#eq(21)
def mf_k(alpha, beta, c):
    return c*(1 + beta/(1-2*beta*c)/(1-beta*c))

#eq(22)
def mf_k2(alpha, beta, c):
    return c*(c + 1/(1-2*beta*c)/(1-beta*c))

#solve for p from eq(18)
def p_eq(p,alpha,beta,N):
    return 0.5*(np.tanh(beta*N*p + alpha)+1)-p

#solve for c from eq(A.2)
def c_eq(c, alpha, beta, N):
    return N*np.exp(2*beta*c+2*alpha)-c

#solve for <k> from eq(55)
def k_eq(kmean,alpha,beta,trunc,N):
    return sum_k(1,kmean,trunc,alpha,beta,N)/sum_k(0,kmean,trunc,alpha,beta,N)-kmean

N = 300
alpha = -0.5-0.5*np.log(N)

# =============================================================================
# #you can check #of solution of (18) graphically here
# =============================================================================
# x=np.arange(0,10,0.1)
# plt.plot(x,[k_eq(i,4,-0.5,,N,N) for i in x])
# plt.plot(x,x)
# plt.show()

# kmean = scpo.bisect(k_eq,0.01,1000,args=(-0.5,-0.8,N,N))
# print(kmean,k_eq(kmean,-0.5,-0.8,500,500))

# =============================================================================
# calculate MF FC
# =============================================================================
# A = np.loadtxt('4alpha = %.2f, N= %d.txt'%(alpha,300))
# k_mf = []
# k2_mf = []
# k_fc = []
# k2_fc = []

# for beta in np.arange(-1,0,0.05):

    # c1 = scpo.bisect(p_eq,-1,1,args=(alpha,beta,N))*N
    # c2 = scpo.newton(c_eq,3,args=(alpha,beta,N)) 
    # print(c1/N)
    # print(mf_k(alpha,beta,c1))
    
    
    # k_mf.append(mf_k(alpha,beta,c1))
    # k2_mf.append(mf_k2(alpha,beta,c1))                
    
    
    # kmean = scpo.newton(k_eq,5,args=(4,beta,N,N))
    # k_fc.append(kmean)
    # k2_fc.append(finite_k2(4,beta,kmean,N,N))

# np.savetxt('mf_alpha = %.2f, N= %d.txt'%(alpha,N),np.transpose(np.concatenate(([A[:,0]],[k_mf],[k2_mf]))))
# np.savetxt('fc_alpha = %.2f, N= %d.txt'%(4,N),np.transpose(np.concatenate(([A[:,0]],[k_fc],[k2_fc]))))
 
# =============================================================================
# load saved data
# =============================================================================
A = np.loadtxt('4alpha = %.2f, N= %d.txt'%(alpha,300))
# A1 = np.loadtxt('3alpha = %.2f, N= %d.txt'%(alpha,N))
k_mf = np.loadtxt('mf_alpha = %.2f, N= %d.txt'%(alpha,N))
k_fc = np.loadtxt('fc_alpha = %.2f, N= %d.txt'%(-0.5,N))

plt.figure(figsize = (8,6))
plt.plot(A[:,0],np.log(A[:,1])/np.log(N),'o',label = 'MCMC1')
# plt.plot(A1[:,0],np.log(A1[:,1]),'o',label = 'MCMC2')
plt.plot(k_mf[:20,0],np.log(k_mf[:20,1])/np.log(N),'o', label = 'MF')
plt.plot(k_fc[:20,0],np.log(k_fc[:20,1])/np.log(N),'o', label = 'FC')
plt.ylabel('log connectivity',fontsize = 14)
plt.xlabel('beta',fontsize = 14)

plt.legend()
plt.grid()
plt.show()