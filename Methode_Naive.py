# -*- coding: utf-8 -*-
"""
Created on Fri May  5 00:52:52 2017

@author: amirb
"""
#############################################################################################################
#----------------------------------Identification des communautées-------------------------------------#
#############################################################################################################

# Benmahjoub Amir , Vincent Ragel

"""

Ce programme estime des graphs, à l'aide de la méthode de Monte Carlo puis calcul, la probabilité que les deux 
premiers vecteurs du cluster 1 soient mal clusterisés

"""


# Cette méthode, avec certains paramètres, permettent de montrer plusieurs exemples dans lesquels les deux premiers 
# sommets ne sont pas dans le bon cluster. Le but de cet algorithme est de tester la méthode sampling implémentée.
#
#
#


import numpy as np
from random import uniform
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import colors
from collections import Counter
import networkx as nx
from scipy import sparse


#On simule un graph G de matrice d'adjacence A à l'aide stochastic bloc model
# V=[v_1,...v_n] : ensemble des points
# C : C[i] correspond au cluster du point V[i] 
#c_in et c_out sont liées aux probabilités du stochastic bloc model  

def Matrice_adj(V,C,c_in,c_out) : 
    Z_in= [0,0]
    Z_out= [0,0]
    n = len(V) #nb de points
    k = np.max(C)+1  #nb de clusters(+1 car on commence à numéroter par 0)
    p_in=c_in/n    # Les proba du stochastic bloc model
    p_out=c_out/n
    B=p_out*np.ones((k,k)) + np.diagflat(np.ones(k)*(p_in-p_out))
    A=np.zeros((n,n))
    for i in range(int(n)):
        for j in range(i+1):
            p = B[int(C[i])][int(C[j])]
            Z= np.random.binomial(1,p)         
            A[i][j]=A[j][i]=Z
        else :
            p = B[int(C[i])][int(C[j])]
            Z= np.random.binomial(1,p)
            A[i][j]=A[j][i]=Z          
    return (A,Z_in,Z_out)
    

#Cette fonction fournie les k vecteurs propres de A 
# associés aux plus grandes valeurs propres sous forme d'un vecteur U(nxk)
            

def kfirstVP(A,k) : 
    B=sparse.csr_matrix(A)
    vals, vecs = sparse.linalg.eigs(B, k=2 , which = 'LR')
    return vecs.real


# Cette fonction fournie la clusterisation du graph de matrice A
# On obtient un tableau T tel que : l'élèment V[i] appartient 
# au clusteur T[i]
    

def clusteringspectral(A,k) :
    U = kfirstVP(A,k)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(U)
    return (kmeans.labels_)
    
# A partir du tableau T décrit ci dessus, cette fonction fournie
# les clusteurs C1,C2. 
    
def PrintCluster(U):
    C1=[];
    C2=[];
    for i in range (len(U)):
        if U[i]==0:
            C1.append(i)
        else:
            C2.append(i)
    return C1,C2

# Cette fonction fournie le tableau des différences croisées entre
#les clustering C1,C2(référence) et B1,B2 (calculés) : 

"""
             C1    C2
           |----|----|
        B1 | E1 | E2 |   E1 : nombre de différences entre C1 et B1
           |----|----|
        B2 | E3 | E4 |
           |----|----| 
"""

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
    
#Cette fonction donne la pureté du clustering c.a.d
# Il s'agit du nombre de points correctement clusterisés/N
def ClusteringPurity(C,B):
    
    C1=PrintCluster(C)[0]
    C2=PrintCluster(C)[1]
    B1=PrintCluster(B)[0]
    B2=PrintCluster(B)[1]
    
    M = Confusionmatrix(C1,C2,B1,B2)
    
    nberreur = min(M[0][0],M[0][1])+min(M[1][0],M[1][1])
    
    nbsimilarity = len(C)-nberreur
    return(float(nbsimilarity)/len(C))
    

def Resultat(A) : 
    
    U = clusteringspectral(A,2) 
    C1=PrintCluster(C)[0]
    C2=PrintCluster(C)[1]
    B1=PrintCluster(U)[0]
    B2=PrintCluster(U)[1]
    M = Confusionmatrix(C1,C2,B1,B2)
    Q = np.argmin(M[0,:])
    i=0
    E = PrintCluster(U)[Q]
    for k in E:
        if V[0]==k : 
           i=i+1
        if V[1]==k : 
           i=i+1 
    if i==2:
        return 0
    else:
        #print("E",E)
        return 1 

    
##########################################################################


### Parametres du probleme
n=50 #nombre de points
k=2    #nombre de cluster
M = 1000
V=np.arange(n)  
C=np.zeros(n)  
nprime=n/2
C[nprime:]=np.ones(n-nprime)
c_in=25.0
c_out=5.0

p_in=c_in/n
p_out=c_out/n



#CONDITION POUR UN CLUSTERING QUI FONCTIONNE : a>>b
print("##########   Methode Naive ######################### ")
print("CONDITION POUR UN CLUSTERING QUI FONCTIONNE : ")
a=c_in-c_out
print (" c_in-c_out =  ", a)
b=np.sqrt(np.log(len(V))*(c_in+c_out)) 
print (" np.sqrt(np.log(len(V))*(c_in+c_out))  =  ", b)
print (" p_in = ", p_in)
print (" p_out = ", p_out)


print("##################  Sampling Généralisé     ####################")
print("Données du problème :")
print("Nombre d'individus : ",n)
print("Nombre de simulations : ",M)
print("C_in = ", c_in,"  C_out = ", c_out)
print("p_in = ", p_in,"  p_out = ", p_out)

Ech=[]

for i in range(M):
    A=Matrice_adj(V,C,c_in,c_out)
    Ech.append(Resultat(A[0]))
               



print( " ")
print( " ")

Proba = np.mean(Ech)
print("###########  Résultats     ################@")
print("Probailité obtenue = " , np.mean(Ech))
print("Ecart type =" , np.std(Ech) )
print("Demi largeur =" , 1.96 * np.std(Ech) / np.sqrt(len(Ech)) )

print("Intervalle de confiance : " ) 
print("[", np.mean(Ech) - 1.96 * np.std(Ech) / np.sqrt(len(Ech)), " , ", np.mean(Ech) + 1.96 * np.std(Ech) / np.sqrt(len(Ech)),"]")





    
