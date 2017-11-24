# -*- coding: utf-8 -*-
"""
Created on Fri May  5 00:52:52 2017

@author: amirb
"""


#############################################################################################################
#----------------------------------Identification des communautées-------------------------------------#
#############################################################################################################

# Benmahjoub Amir , Vincent Ragel



"""----------------------------Stochastic Block Model et Clustering--------------------------------"""

"""Ce programme simule des graphs à l'aide du Stochastic Bloc Model , calcul le clustering de ces
 graphs puis estime la fiabilité du clustering associé"""

 
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

On simule un graph G de matrice d'adjacence A à l'aide du stochastic bloc model : 

     * V=[v_1,...v_n] : ensemble des points
     * C : C[i] correspond au cluster du point V[i] 
     * c_in et c_out sont liées aux probabilités du stochastic bloc model
     
""" 


def stocha(V,C,c_in,c_out) :  
    
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
    
  
""" Cette fonction fournie la clusterisation du graph de matrice A. On obtient un tableau T tel que : l'élèment
 V[i] appartient  au clusteur T[i] """
    

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
    
def PrintCluster(U):
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

def Confusionmatrix(C1,C2,B1,B2):
    
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
    
    
    
    
"""Cette fonction donne le taux de réussite du clustering.
 Il s'agit de :  #Bienclusterisés/n  """
    
    
def tauxreussite(C,B):
    
    C1=PrintCluster(C)[0]
    C2=PrintCluster(C)[1]
    B1=PrintCluster(B)[0]
    B2=PrintCluster(B)[1]
    
    M = Confusionmatrix(C1,C2,B1,B2)
    
    # la fonction min sur une ligne de B nous permet d'associé un cluster calculé à 
    # son cluster de référence. En effet, il arrive que la méthode k_means switch les 0 et les 1 désignant
    # les clusters. C'est la l'interet de la matrice confusion ici. 
    
    nberreur = min(M[0][0],M[0][1])+min(M[1][0],M[1][1])
    nbsimilarity = len(C)-nberreur
    return(float(nbsimilarity)/len(C))
    
    

########################################################################################################################

#Données du problème

n=100#nombre de points
V=np.arange(n)  

# On crée le tableau qui assigne chaque point de V à un cluster
C=np.zeros(n)   
nprime=n/2# A partir de nprime on met que des 1. 
C[nprime:]=np.ones(n-nprime)


c_in=50.0
c_out=10.0


#CONDITION POUR UN CLUSTERING QUI FONCTIONNE : a>>b

print " "
a=c_in-c_out
print " c_in-c_out =  ", a
print " "
b=np.sqrt(np.log(len(V))*(c_in+c_out)) 
print " np.sqrt(np.log(len(V))*(c_in+c_out))  ", b
print " "

alpha = a/b

print "rapport = ", alpha



#Stochastic Bloc Model simulation 


A=stocha(V,C,c_in,c_out) 



# On calcule la matrice M correspondant à l'option 2. 

D=np.diag(np.ones(n))
for i in range(n):
    D[i][i]= 1.0 / np.sqrt(np.sum(A[i,:]))
    
M = np.dot(np.dot(D,A),D)

# On gère le cas ou np.sqrt(np.sum(A[i,:])) est nul.
        
while  np.isnan(M).any():
    A=stocha(V,C,c_in,c_out) 
    D=np.diag(np.ones(n))
    for i in range(n):
        D[i][i]= 1.0 / np.sqrt(np.sum(A[i,:]))
    M = np.dot(np.dot(D,A),D)
    
    
# Tracé des valeurs propres dans un plan 2D


#B = kfirstVP(A)
#
#Bp = kfirstVP(M)
#
#
#U = clusteringspectral(M)
#B1=np.ones((Counter(U)[0],2))
#B2=np.ones((Counter(U)[1],2))
#
#j=0
#k=0
#
#for i in range (n) : 
#    
#    
#    if U[i]==0:
#        B1[j] = Bp[i]
#        j = j +1 
#    else : 
#        B2[k] = Bp[i]
#        k = k + 1
#        
#fig = plt.figure()
#fig.suptitle('X2 = f(X1)', fontsize=14, fontweight='bold')
#
#ax = fig.add_subplot(111)
#fig.subplots_adjust(top=0.85)
#
#ax.set_xlabel('X1')
#ax.set_ylabel('X2')
#        
#plt.scatter(B1[:,0], B1[:,1], c = 'red')
#plt.scatter(B2[:,0], B2[:,1], c = 'yellow')
#
#
#
#plt.savefig("COMPA3")


    
"""----------------------------------Affichage des graphs-------------------------------------------"""
    

#print "La matrice d'adjacence est : "
#print(A)

#Affichage de A
#
#cmap = colors.ListedColormap(['white', 'red'])
#bounds=[0,0.5,1]
#norm = colors.BoundaryNorm(bounds, cmap.N)
#
#red_patch = mpatches.Patch(color='red', label="existence d'un lien")
#plt.legend(handles=[red_patch])
#
#img = plt.imshow(A, interpolation='nearest', 
#                    cmap=cmap, norm=norm)
#plt.title("Matrice d'adjacence (c_in,c_out) = (12,10) ")
#
#plt.savefig("Matriceadj_cin10_cout5")
#
#plt.show()



#Affichage du Graph G 
#


#G=nx.Graph()
#G.add_nodes_from(V)
#
#for i in range(n):
#    for j in range(i+1):
#        if (A[i][j]==1) :
#            G.add_edge(i,j)
#plt.figure()            
#nx.draw(G)
#
#plt.savefig("Couverture") 




#Affichage du taux de réussite 


#Clusteringspectrale option 1

U = clusteringspectral(A)

#Clusteringspectrale option 2

U2 = clusteringspectral(M)

# Pureté 

print "Taux de réussite option 1 : ", tauxreussite(C,U)
print "Taux de réussite option 2 : " ,tauxreussite(C,U2)























    
    


    
    
    
    
   
    
    
    


    
    
    
    
    
    
    









