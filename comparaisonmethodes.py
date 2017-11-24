# -*- coding: utf-8 -*-
"""
Created on Fri May  5 00:52:52 2017

@author: amirb
"""
#############################################################################################################
#----------------------------------Identification des communautées-------------------------------------#
#############################################################################################################

# Benmahjoub Amir , Vincent Ragel


"""----------------------------tauxdereussite = f(taille du cluster 1)--------------------------------"""

"""

Ce programme permet de tracer la courbe tauxdereussite = f(taille du cluster 1) pour chacune des deux
options puis de les comparer 

Les programmes non commentés le sont déjà sur StochaBM.py 

"""



import numpy as np
from random import uniform
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import colors
from collections import Counter





def stocha(V,C,c_in,c_out) :   
    n = len(V) 
    k = 2
    p_in=c_in/n    
    p_out=c_out/n
    B=p_out*np.ones((k,k)) + np.diagflat(np.ones(k)*(p_in-p_out)) 
    A=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            p = B[int(C[i])][int(C[j])]
            A[i][j]=A[j][i]=np.random.binomial(1,p)
                
                
    return A
                

def kfirstVP(A) : 
    n= len(A[0])
    U=np.zeros((n,2))
    V = np.linalg.eig(A) 
    Valeurspropres= V[0]
    Vecteurspropres = V[1]
    Valeursproprestriee=np.sort(Valeurspropres)
    N=len(Valeursproprestriee)
    for i in range(2):
        VP = Valeursproprestriee[N-i-1]
        i1 = Valeurspropres.tolist().index(VP)
        U[:,i]= Vecteurspropres[:,i1]
    return U 


def clusteringspectral(A) :
    U = kfirstVP(A)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(U)
    return (kmeans.labels_)
    

def PrintCluster(U):
    C1=[];
    C2=[];
    for i in range (len(U)):
        if U[i]==0:
            C1.append(i)
        else:
            C2.append(i)
    return C1,C2



def Confusionmatrix(C1,C2,B1,B2):
    
    M =np.ones((2,2))
    
    M_0_0 = Counter(C1) - Counter(B1)
    M_0_0b = Counter(B1) - Counter(C1)
    
    M_0_1 = Counter(B1) - Counter(C2)
    M_0_1b = Counter(C2) - Counter(B1)

    M_1_0 = Counter(C1) - Counter(B2)
    M_1_0b = Counter(B2) - Counter(C1)
    
    M_1_1 = Counter(B2) - Counter(C2)
    M_1_1b = Counter(C2) - Counter(B2)
    
    M[0][0]=len(M_0_0)+len(M_0_0b)
    M[0][1]=len(M_0_1)+len(M_0_1b)
    M[1][0]=len(M_1_0)+len(M_1_0b)
    M[1][1]=len(M_1_1)+len(M_1_1b)
    
    return(M)
    

def tauxreussite(C,B):
    
    C1=PrintCluster(C)[0]
    C2=PrintCluster(C)[1]
    B1=PrintCluster(B)[0]
    B2=PrintCluster(B)[1]
    
    M = Confusionmatrix(C1,C2,B1,B2)
    
    nberreur = min(M[0][0],M[0][1])+min(M[1][0],M[1][1])
    nbsimilarity = len(C)-nberreur
    return(float(nbsimilarity)/len(C))

    
########################################################################################################################

# Données du problème 


n=100 
V=np.arange(n)  
C=np.zeros(n)
      
c_in=50.0
c_out=10.0


taillecluster1 = []
tauxreussite1=[]
tauxreussite2=[]

nprime=0
C[nprime:]=np.ones(n-nprime)

for i in range (100):
    
    print(i)
    
    taillecluster1.append(nprime)

    p1=0
    p2=0
    
    for j in range(50): # On moyenne sur 50 valeurs
        #Methode 1
        A=stocha(V,C,c_in,c_out)
        #Methode 2
        D=np.diag(np.ones(n))
        
        for i in range(n):
            D[i][i]= 1.0 / np.sqrt(np.sum(A[i,:]))
            M = np.dot(np.dot(D,A),D)
        
        
        while  np.isnan(M).any():
            
            A=stocha(V,C,c_in,c_out) 
            D=np.diag(np.ones(n))   
            for i in range(n):
                 D[i]= 1.0 / np.sqrt(np.sum(A[i,:]))
            M = np.dot(np.dot(D,A),D)
            
        #Calcul des taux de réussite pour les deux méthodes
        U = clusteringspectral(A)
        U2 = clusteringspectral(M)
        
        p1=p1+tauxreussite(C,U)
        p2=p2+tauxreussite(C,U2)
    #on récupère les moyennes associées   
    tauxreussite1.append(p1/50.0)
    
    tauxreussite2.append(p2/50.0)
        
    #on augmente la taille du cluster 1 et on recrée C. 
    nprime=nprime + 1
    C=np.zeros(n) 
    C[nprime:]=np.ones(n-nprime)
    
    
fig = plt.figure()
fig.suptitle('tauxdereussite = f(taille du cluster 1)', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)


ax.set_xlabel('taille cluster 1')
ax.set_ylabel('taux de reussite')

        
plt.plot(taillecluster1,tauxreussite1,label="Algorithme 2")
plt.plot(taillecluster1,tauxreussite2,label="Algorithme 3")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.savefig('compaalgo1.png')    

plt.show()

   
        
        
    
    
    
    
    
   
        
          
          
          
    
    

   
  
    
    
    
    





























    
    


    
    
    
    
   
    
    
    


    
    
    
    
    
    
    









