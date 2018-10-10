import numpy as np

steps = 5000        #最大迭代步数
alpha  = 0.0002     #学习率
beta = 0.02         #控制特征向量，避免出现非常大的值                                

def matrix_factorisation(R, P, Q, K, steps= 5000, alpha= 0.0002, beta= 0.02):
    Q = Q.T 
    for st in range(steps):
        #这里注意为何要双循环，实际上对每个R中的值都要求误差
        for i in range(len(R)):
            for j in range (len(R[i])):
                if R[i,j] > 0:       #保证R没有负值
                    eij = R[i, j] - np.dot(P[i,:], Q[:,j])
                    for k in range(K):
                        P[i,k] = P[i,k] + alpha * (2 * eij * Q[k,j] - beta * P[i,k])
                        Q[k,j] = Q[k,j] + alpha * (2 * eij * P[i,k] - beta * Q[k,j])
        eR = np.dot(P,Q)             #P*Q的实际值
        e = 0
        for i in range(len(R)):
            for j in range(len(R[i])):
                if R[i,j] > 0:
                    e = e + pow(R[i,j] - eR[i,j], 2)
                    for k in range(K):
                        e = e + (beta / 2) * (pow(P[i,k],2) + pow(Q[k,j],2))
        if e < 0.001:
            break
    return P, Q.T

 
R = [[5,3,0,1], [4,0,0,1], [1,1,0,5],[1,0,0,4],[0,1,5,4]]     
R = np.array(R)  
   
N = len(R)  
M = len(R[0])  
K = 2  
   
P = np.random.rand(N,K)  
Q = np.random.rand(M,K)  
   
nP, nQ = matrix_factorisation(R, P, Q, K)  
nR = np.dot(nP, nQ.T) 

print(nR)
