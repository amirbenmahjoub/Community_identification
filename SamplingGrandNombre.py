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

Ce programme estime , à l’aide de l’importance sampling, la probabilité qu’un ensemble
de grand cardinal (plus de 40 pourcents des individus), soit mal clusterisé    

"""

"""Le sampling ne modifie que toutes les probas"""

import numpy as np
from clustering import matrice_Adjacence_Sampling_Q2_2
from clustering import indicatrice_grdNombres_Q22

    



 
## Données du problème : 
n=50 #nombre de points
k=2  #nombre de cluster
M = 1000
 
c_in=19.0
c_out=8.0
p_in=c_in/n
p_out=c_out/n

c_in_prime=12.0
c_out_prime=9.0
p_in_prime=c_in_prime/n
p_out_prime=c_out_prime/n


### Paramètres de la simulation
p1=2 ##Nombre de points à considérer dans le cluster n1
p2=0 ##Nombre de points à considérer dans le cluster n1



    
V=np.arange(n)  
C=np.zeros(n)
#Nprime définit le nombre de 1
nprime=n/2
C[nprime:]=np.ones(n-nprime)  



### Coeffiicent multiplicateur
S_in = (1-p_in)/(1-p_in_prime)
S_in_simple = p_in/p_in_prime
S_out = (1-p_out)/(1-p_out_prime)
S_out_simple = (p_out)/(p_out_prime)

               
F_in = (p_in*(1-p_in_prime))/(p_in_prime*(1-p_in))
F_out = (p_out*(1-p_out_prime))/(p_out_prime*(1-p_out))


print("##################  Sampling Généralisé     ####################")
print("Données du problème :")
print("Nombre d'individus : ",n)
print("Nombre de simulations : ",M)
print("C_in = ", c_in,"  C_out = ", c_out)
print("p_in = ", p_in,"  p_out = ", p_out)
print("C_in_prime = ", c_in_prime,"  C_out_prime = ", c_out_prime)
print("p_in_prime = ", p_in_prime,"  p_out_prime = ", p_out_prime)


print(" ")
print("Condition pour un clustering fonctionnel : ")
print(" c_in - c_out > np.sqrt(np.log(len(V))*(c_in+c_out)) ")




#CONDITION POUR UN CLUSTERING QUI FONCTIONNE : a>>b

a=c_in-c_out
b=np.sqrt(np.log(len(V))*(c_in+c_out))
if (a>b):
    print( a , " >   ", b, "   Condition vérifée")
else:
    print( a , " <  ", b, "   Condition non vérifée")

#print (" c_in-c_out =  ", a, " np.sqrt(np.log(len(V))*(c_in+c_out))  =  ", b)

a=c_in_prime-c_out_prime
b=np.sqrt(np.log(len(V))*(c_in_prime+c_out_prime)) 
if (a>b):
    print( a , " >   ", b, "   Condition vérifée")
else:
    print( a , " <  ", b, "   Condition non vérifée")
    
print( " ")
#print (" c_in_prime-c_out_prime =  ", a , " np.sqrt(np.log(len(V))*(c_in_prime+c_out_prime))  =  ", b)

#print("Coefficients  :::")
#print("S_in = (1-p_in)/(1-p_in_prime)",S_in)
#print("S_in_simple = p_in/p_in_prime",S_in_simple)
#print("S_out = (1-p_out)/(1-p_out_prime)",S_out)
#print("S_out_simple = (1-p_out)/(1-p_out_prime)", S_out_simple)


print("F_in",F_in)
print("F_out",F_out)

Ech=[]

for i in range(M):
    # On réalise M clustering avec les paramètres énoncés ci-dessus
    A=matrice_Adjacence_Sampling_Q2_2(V,C,c_in,c_out,c_in_prime,c_out_prime)
    Z_in = A[1] #Cardinal de Zin=1
    Z_out = A[2]  #Cardinal de Zout=1
    Ech.append(indicatrice_grdNombres_Q22(A[0],C,V,nprime)*(F_in**(np.sum(Z_in)))*(F_out**(np.sum(Z_out))))
    


### Ces deux lignes sont étranges 
nbcouplesamecluster = len(Z_in)
nbcoupleclustediff = len(Z_out)

#Debug
#print("nbcouplesamecluster",nbcouplesamecluster)
#print("nbcoupleclustediff",nbcoupleclustediff)

P = (((1- p_in)/(1-p_in_prime))**nbcouplesamecluster) * (((1- p_out) / (1- p_out_prime))**(nbcoupleclustediff))
Ech = P*np.array(Ech)

Proba = np.mean(Ech)



print( " ")
print( " ")

Proba = np.mean(Ech)
print("###########  Résultats     ################@")
print("Probailité obtenue = " , np.mean(Ech))
print("Ecart type =" , np.std(Ech) )
print("Demi largeur =" , 1.96 * np.std(Ech) / np.sqrt(len(Ech)) )

print("Intervalle de confiance : " ) 
print("[", np.mean(Ech) - 1.96 * np.std(Ech) / np.sqrt(len(Ech)), " , ", np.mean(Ech) + 1.96 * np.std(Ech) / np.sqrt(len(Ech)),"]")















