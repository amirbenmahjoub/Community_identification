#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Projet de Modal :  Amir Benmahjoub, Vincent Ragel


#############################################################################################################
#----------------------------------Identification des communautées-------------------------------------#
#############################################################################################################
"""

"""----------------------------Stochastic Block Model et Clustering--------------------------------"""



"""
Le fichier clustering.py récupère les fonctions nécessaires aux méthodes Sampling Général (ie la méthode de la question
2.1) et Sampling Grand Nombre.

Nous avons essayé d'expliquer la fonction et l'utiliter de chaque fonction dans ce fichier.Toutefois
ce fichier vient après en fin de période. Par conséquent certains fonctions sont indépendantes dans 
leur propre fichier.

Globalement les fonctions matrices_d'adjacence permettent de générer une matrice d'adjacence en fonction 
des paramètres initiaux (généralement n, nprime, C_in, C_out, etc..). En fonction de la fonction certaines 
fonctions possèdent des besoins différents c'est pour cette raison que les noms possède un suffixe qui indique
pour quelle fonction elles ont été désignées.


Ensuite les fonctions indicatrices permettent de vérifier si la condition de clusterisation est bien vérifiée.
Elles dépendent également du numéro de la question.


Ce programme simule des graphs à l'aide du Stochastic Bloc Model , calcul le clustering de ces
graphs puis estime la fiabilité du clustering associé
 
 
 
 """

 
import numpy as np
from random import uniform
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import colors
from collections import Counter
import networkx as nx
from scipy import sparse
import time
import matplotlib.patches as mpatches


""" 
Q1.

On simule un graph G de matrice d'adjacence A à l'aide du stochastic bloc model : 

     * V=[v_1,...v_n] : ensemble des points
     * C : C[i] correspond au cluster du point V[i] 
     * c_in et c_out sont liées aux probabilités du stochastic bloc model
     
""" 


def matrice_Adjacence(V,C,c_in,c_out) :  
    
    n = len(V) #nb de points
    k = 2  #nb de clusters = 2
    p_in=c_in/n    # Les proba du stochastic bloc model
    p_out=c_out/n
    B= p_out*np.ones((k,k)) + np.diagflat(np.ones(k)*(p_in-p_out)) #Matrice du stochastic bloc model
    
    #Construction de la matrice d'adjacence
    A=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            p = B[int(C[i])][int(C[j])]
            A[i][j]=A[j][i]=np.random.binomial(1,p) #Graph indirect, matrice symétrique. 
                
    return A
    


""" 
Q2.1

Sachant que les évènements étudiés ont une probabilité rare d'apparition nous allons utilisé la
méthode sampling pour augmenter la probabiltié d'apparition de ces évènements et ainsi pour calculer 
l'espérance de. 

Contrairement à la fonction précédente nous allons modifier la probabilité des p1 premiers élèments du cluster 
1 et la probabilité des p2 élèments du cluster 2.

Les vecteurs Z_in et Z_out permettent de calculer le cardinal des Zin=1 et Zin=0. Ces paramètres sont utilisés 
dans la formule de changement de loi
     
""" 



def matrice_Adjacence_Sampling_Q2_1(V,C,nprime,c_in,c_out,c_in_prime,c_out_prime,p1,p2) : 
    ## Les listes sont créées pour différencier les individus
    list1 = []
    list2 = []
    for h in range(p1):
        list1.append(h)
    n0 = int(nprime)
    n1 = int(nprime+p2)
    for h in range(n0,n1):
        list2.append(h)
    Z_in= [0,0]
    Z_out= [0,0]
    n = len(V) #nb de points
    k = np.max(C)+1  #nb de clusters(+1 car on commence à numéroter par 0)
    p_in=c_in/n    # Les proba du stochastic bloc model
    p_out=c_out/n
    p_in_prime=c_in_prime/n   
    p_out_prime=c_out_prime/n
    B=p_out*np.ones((k,k)) + np.diagflat(np.ones(k)*(p_in-p_out))
    D_prime=p_out_prime*np.ones((k,k)) + np.diagflat(np.ones(k)*(p_in_prime-p_out_prime))

    A=np.zeros((n,n))
    for i in range(int(n)):
        for j in range(i+1):
            if( (j in list1) | (j in list2) ):
                p = D_prime[int(C[i])][int(C[j])]
                Z= np.random.binomial(1,p)
                # CAS => un lien est cree
                if (C[i]!=C[j]):
                    if Z == 1:
                        Z_out[1] +=1
                    else :
                        Z_out[0] +=1                      
                else:
                    if Z == 1:
                        Z_in[1] +=1
                    else :
                        Z_in[0] +=1
                #Creation du lien dans la matrix adjency              
                A[i][j]=A[j][i]=Z
            else :
                p = B[int(C[i])][int(C[j])]
                Z= np.random.binomial(1,p)
                A[i][j]=A[j][i]=Z
               
    return (A,Z_in,Z_out)
  


""" 
Sauvegarde


"""


def g(A,C,V,nprime) : 
    
    U = clusteringspectral(A) 
    C1=printCluster(C)[0]
    C2=printCluster(C)[1]
    B1=printCluster(U)[0]
    B2=printCluster(U)[1]
    M = confusionmatrix(C1,C2,B1,B2)
    i=0
    ClusterassocieC1 = printCluster(U)[np.argmin(M[0,:])]
    ClusterassocieC2 = printCluster(U)[np.argmax(M[0,:])]
    
    for k in V[:nprime] : 
        
        if k not in ClusterassocieC1:
            i=i+1
    
    for k in V[nprime:] :
        
        if k not in ClusterassocieC2:
            
            i=i+1
           
    if i>(len(A[0])*0.4):
        
        return 1
        
    else:
        return 0




""" 
Q2.2

Cette fonction crée une matrice d'adjacence en fonction des probabilités p_in_prime et p_out_prime.

Contrairement à la fonction précédente les probabilités de créer des liens entre les individus sont 
modifiées pour tous les individus.

 """


def matrice_Adjacence_Sampling_Q2_2(V,C,c_in,c_out,c_in_prime,c_out_prime) : 
    Z_in=[]
    Z_out=[]
    n = len(V) #nb de points
    k = np.max(C)+1  #nb de clusters(+1 car on commence Ã  numÃ©roter par 0)
    p_in_prime=c_in_prime/n   
    p_out_prime=c_out_prime/n
    D_prime=p_out_prime*np.ones((k,k)) + np.diagflat(np.ones(k)*(p_in_prime-p_out_prime))
    #Construction de la matrice d'adjacence
    A=np.zeros((n,n))
    for i in range(int(n)):
        for j in range(i+1):
            
                p = D_prime[int(C[i])][int(C[j])]
                Z= np.random.binomial(1,p)
                
                if p==p_in_prime:
                    Z_in.append(Z)
                    #Z_in.append(Z)  Précédemment attention pour le changement
                else : 
                    Z_out.append(Z)
                    
                A[i][j]=A[j][i]=Z
                 
               
    return (A,Z_in,Z_out)

""" 
Utile 

Cette fonction fournie les deux vecteurs propres de A associés aux plus grandes valeurs propres 
sous forme d'un vecteur U(nxk). On utilise la structure de matrice creuse à l'aide de sparse.csr_matrix puisque
notre matrice contient un nombre important de 0 ce qui augmente la rapidité du calcul. On utilise également
une fonction linal.eigsh qui calcule les valeurs propres spécialement pour les matrices symétriques ce qui 
est notre cas. Notons que d'après le théorème spéctrale toutes les valeurs propres et vecteurs propres sont réels.


 """
            

def kfirstVP(A) : 
    B=sparse.csr_matrix(A) #On transforme en matrice creuse
    vals, vecs = sparse.linalg.eigsh(B, k=2 , which = 'LA') # On récupère seulement les deux plus grandes
    return vecs.real #On prend la partie réel pour s'assurer de certains bug
    
  
""" 

Cette fonction fournie la clusterisation du graph de matrice A. On obtient un tableau T tel que : l'élèment
 V[i] appartient  au clusteur T[i] 
 
 """
    

def clusteringspectral(A) :
    U = kfirstVP(A)  #On récupère les vecteurs propres associés aux deux plus grandes valeurs propres de A
    kmeans = KMeans(n_clusters=2, random_state=0).fit(U) # On effectue un K-means clustering
    return (kmeans.labels_) #On récupère le tableau T décrit ci dessus
    
"""


A partir du tableau T décrit ci dessus, cette fonction fournie
les clusteurs C1,C2. Exemple : 

T = [0,0,0,1,1,1]

PrintCluster(U)= [[0,1,2],[3,4,5]]


""" 
    
def printCluster(U):
    C1=[]; # Cluster 1 associé à la valeur 0
    C2=[]; # Cluster 2 associé à 1a valeur 1
    for i in range (len(U)):
        if U[i]==0:
            C1.append(i)  
        else:
            C2.append(i)
    return C1,C2 

"""

Cette fonction fournie le tableau (2X2) des différences croisées entre

les clustering C1,C2(référence) et B1,B2 (calculés) : 



             C1    C2
           |----|----|
        B1 | E1 | E2 |   Ex : E1 : nombre de différences entre C1 et B1
           |----|----|   
        B1 | E3 | E4 |
           |----|----| 

Ce tableau permet dans ce code d'estimer  après la fiabilité du clustering. Nous verrons
dans d'autres codes qu'il nous sera également très utile. 


"""

def confusionmatrix(C1,C2,B1,B2):
    
    M =np.ones((2,2))
    
    #Calcul des différences en C1 et B1
    M_0_0 = Counter(C1) - Counter(B1)
    M_0_0b = Counter(B1) - Counter(C1)
    
    #Calcul des différences en C2 et B1
    M_0_1 = Counter(B1) - Counter(C2)
    M_0_1b = Counter(C2) - Counter(B1)

    #Calcul des différences en C1 et B2
    M_1_0 = Counter(C1) - Counter(B2)
    M_1_0b = Counter(B2) - Counter(C1)
    
    #Calcul des différences en C2 et B2    
    M_1_1 = Counter(B2) - Counter(C2)
    M_1_1b = Counter(C2) - Counter(B2)
    
    M[0][0]=len(M_0_0)+len(M_0_0b)
    M[0][1]=len(M_0_1)+len(M_0_1b)
    M[1][0]=len(M_1_0)+len(M_1_0b)
    M[1][1]=len(M_1_1)+len(M_1_1b)
    
    return(M)
    


def clusteringPurity(C,B):
    
    C1=printCluster(C)[0]
    C2=printCluster(C)[1]
    B1=printCluster(B)[0]
    B2=printCluster(B)[1]
    
    M = confusionmatrix(C1,C2,B1,B2)
    
    nberreur = min(M[0][0],M[0][1])+min(M[1][0],M[1][1])
    
    nbsimilarity = len(C)-nberreur
    return(float(nbsimilarity)/len(C))

    
    
"""Cette fonction donne le taux de réussite du clustering.
 Il s'agit de :  #Bienclusterisés/n  """
    
    
def tauxreussite(C,B):
    
    C1=printCluster(C)[0]
    C2=printCluster(C)[1]
    B1=printCluster(B)[0]
    B2=printCluster(B)[1]
    
    M = confusionmatrix(C1,C2,B1,B2)
    
    # la fonction min sur une ligne de B nous permet d'associé un cluster calculé à 
    # son cluster de référence. En effet, il arrive que la méthode k_means switch les 0 et les 1 désignant
    # les clusters. C'est la l'interet de la matrice confusion ici. 
    
    nberreur = min(M[0][0],M[0][1])+min(M[1][0],M[1][1])
    nbsimilarity = len(C)-nberreur
    return(float(nbsimilarity)/len(C))
    
    
""" 
Q2.1

La fonction indicatrice General Q21 permet de prendre en compte les caractéristiques spécifiques à la 
question Q2.1. Nous avons pris en compte la possibilité de prendre en compte p1 élèments du cluster 1 et 
p2 élèments du cluster 2. Il faut donc maintenant vérifier si ces deux vecteurs sont bien clusterisés.

     
""" 





def indicatrice_General_Q2_1(A,C,p1,p2,nprime) : 
    list1 = []
    list2 = []
    for h in range(p1):
        list1.append(h)
    n0 = int(nprime)
    n1 = int(nprime+p2)
    for h in range(n0,n1):
        list2.append(h)
    U = clusteringspectral(A) 
    C1=printCluster(C)[0]
    C2=printCluster(C)[1]
    B1=printCluster(U)[0]
    B2=printCluster(U)[1]
    M = confusionmatrix(C1,C2,B1,B2)
    i2=0
    h1=0
    ClusterassocieC1 = printCluster(U)[np.argmin(M[0,:])]
    ClusterassocieC2 = printCluster(U)[np.argmax(M[0,:])]
    
    for k in list1:
        if k in ClusterassocieC1:
            h1=h1+1
            
    for k in list2:
        if k in ClusterassocieC2:
            i2=i2+1

    if (h1==p1)&(i2==p2):
        return 0   #Return 0 si les vecteurs sont bien clusterisés
    else:
        return 1  #Return 1 si un des vecteurs est mal clusterisé


    
""" 
Q2.2

Dorénavant nous souhaitons savoir si un grand nombre d'individus est mal clusterisé. 

Nous pouvons jouer sur le pourcentage d'individus mal cluster. Il est initialement mis à 40%.



     
""" 

def indicatrice_grdNombres_Q22(A,C,V,nprime) : 
    
    U = clusteringspectral(A) 
    C1=printCluster(C)[0]
    C2=printCluster(C)[1]
    B1=printCluster(U)[0]
    B2=printCluster(U)[1]
    M = confusionmatrix(C1,C2,B1,B2)
    i=0
    ClusterassocieC1 = printCluster(U)[np.argmin(M[0,:])]
    ClusterassocieC2 = printCluster(U)[np.argmax(M[0,:])]
    
    for k in V[:nprime] : 
        if k not in ClusterassocieC1:
            i=i+1
    for k in V[nprime:] :
        if k not in ClusterassocieC2:
            i=i+1 
    if i>(len(A[0])*0.4):
        return 1  #Return 1 si un grand nombre de vecteurs est mal clusterisé
    else:
        return 0




########################################################################################################################


