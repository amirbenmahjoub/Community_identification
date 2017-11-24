# -*- coding: utf-8 -*-
"""
Created on Fri May  5 00:52:52 2017

@author: amirb
"""
#############################################################################################################
#----------------------------------Identification des communautées-------------------------------------#
#############################################################################################################

# Benmahjoub Amir , Vincent Ragel

"""Ce programme trace le taux de réussite en fonction du coefficient alpha qui est
le rapport entre c_in-c_out  et np.sqrt(np.log(len(V))*(c_in+c_out)) """



import numpy as np
from random import uniform
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib import colors
from collections import Counter
import networkx as nx



def stocha(V,C,c_in,c_out) :   
    n = len(V) #nb de points
    k = np.max(C)+1  #nb de clusters(+1 car on commence à numéroter par 0)
    p_in=c_in/n    # Les proba du stochastic bloc model
    p_out=c_out/n
    B=p_out*np.ones((k,k)) + np.diagflat(np.ones(k)*(p_in-p_out)) #Matrice du stochastic bloc model
    
    #Construction de la matrice d'adjacence
    A=np.zeros((n,n))
    for i in range(n):
        for j in range(i+1):
            p = B[int(C[i])][int(C[j])]
            A[i][j]=A[j][i]=np.random.binomial(1,p)
                
                
    return A
    

#Cette fonction fournie les k vecteurs propres de A 
# associés aux plus grandes valeurs propres sous forme d'un vecteur U(nxk)
            

def kfirstVP(A,k) : 
    n= len(A[0])
    U=np.zeros((n,k))
    V = np.linalg.eig(A) 
    Valeurspropres= V[0]
    Vecteurspropres = V[1]
    Valeursproprestriee=np.sort(Valeurspropres)
    N=len(Valeursproprestriee)
    for i in range(k):
        VP = Valeursproprestriee[N-i-1]
        i1 = Valeurspropres.tolist().index(VP)
        U[:,i]= Vecteurspropres[:,i1]
    return U 
  

# Cette fonction fournie la clusterisation du graph de matrice A
# On obtient un tableau T tel que : l'élèment V[i] appartient 
# au clusteur T[i]
    

def clusteringspectral(A,k) :
    U = kfirstVP(A,k)
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
    

def ClusteringPurity(C,B):
    
    C1=PrintCluster(C)[0]
    C2=PrintCluster(C)[1]
    B1=PrintCluster(B)[0]
    B2=PrintCluster(B)[1]
    
    M = Confusionmatrix(C1,C2,B1,B2)
    
    nberreur = min(M[0][0],M[0][1])+min(M[1][0],M[1][1])
    nbsimilarity = len(C)-nberreur
    return(float(nbsimilarity)/len(C))

    

############################################################################################################
      
#Données du problème 
      
      
n1=100
#n2 = 200.0

k=2    


V1=np.arange(n1)  
C1=np.zeros(n1)   
nprime1=n1/2
C1[nprime1:]=np.ones(n1-nprime1)


#V2=np.arange(n2)  
#C2=np.zeros(n2)   
#nprime2=n2/2
#C2[nprime2:]=np.ones(n2-nprime2)

#on part de alpha= 0
c_in=20.0
c_out=20.0
tablalpha=[]
purity=[]
#purityp=[]

#c_in2=n2/2.0
#c_out2=n2/2.0
#tablalpha2=[]
#purity2=[]






for i in range (20): #On trace 20 points
    
    print(i)
    #calcule de alpha
    a=c_in-c_out
    b=np.sqrt(np.log(len(V1))*(c_in+c_out)) 
    tablalpha.append(a/b)
    
    
    #a2=c_in2-c_out2
    #b2=np.sqrt(np.log(len(V2))*(c_in2+c_out2)) 
   # tablalpha2.append(a2/b2)
    

       
    
    Pu1 = 0.0
   #Puprime = 0.0
   # Pu2 = 0.0
    
    for j in range(60) : #On moyenne sur 60 valeurs pour lisser la courbe. 
        
        
        A1=stocha(V1,C1,c_in,c_out)
        
#        D=np.diag(np.ones(n1))
#        for i in range(n1):
#            D[i]= 1.0 / np.sqrt(np.sum(A1[i,:]))
#        M = np.dot(np.dot(D,A1),D)
#
#        while  np.isnan(M).any():
#            A=stocha(V1,C1,c_in,c_out) 
#            D=np.diag(np.ones(n1))
#            for i in range(n1):
#                D[i]= 1.0 / np.sqrt(np.sum(A1[i,:]))
#            M =np.dot(np.dot(D,A1),D)
         
         
         
       # A2=stocha(V2,C2,c_in2,c_out2)
         
         #calcul de la pureté
        U1 = clusteringspectral(A1,k)
        Pu1 = Pu1 + ClusteringPurity(C1,U1)
        #Up = clusteringspectral(M,k)
        #Puprime = Puprime + ClusteringPurity(C1,Up)
        
       # U2 = clusteringspectral(A2,k)
       # Pu2 = Pu2 + ClusteringPurity(C2,U2)
        
        
    
    purity.append(Pu1/60.0) # on moyenne
   # purityp.append(Puprime/60.0)
   # purity2.append(Pu2/60.0)
    
    #on modifie le rapport pour augmenter alpha
    c_in=c_in+1 
    c_out=c_out-1
    
   # c_in2=c_in2+1
   # c_out2=c_out2-1
    
fig = plt.figure()
fig.suptitle('tauxdereussite = f(alpha)', fontsize=14, fontweight='bold')

ax = fig.add_subplot(111)
fig.subplots_adjust(top=0.85)


ax.set_xlabel('alpha')
ax.set_ylabel('taux de reussite')

    
    
plt.plot(tablalpha,purity,label="Algo2")
#plt.plot(tablalpha,purityp,label="Algo3")
#plt.plot(tablalpha2,purity2,label="n=200")

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

plt.savefig("tauxdereussite2Bis.png")



plt.show()



























    
    


    
    
    
    
   
    
    
    


    
    
    
    
    
    
    









