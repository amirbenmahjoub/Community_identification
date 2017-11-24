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


Ce programme estime les meilleurs paramètres C_in_prime et C_out_prime.. Pour les déterminer nous 
rapprochons les valeurs C_in_prime et C_out_prime et simulons M fois le clustering (et on vérifie si les deux 
élèments sont bien clusterisés).

Nous multiplions cette donnée au carré afin de calculer l'espérance au carré. L'espérance au carré pour ces
M simulations est comparé à l'EspéranceMinimale. Si elle est plus faible nous avons trouvé un nouveau couple
meilleur que le précédent. Sinon on ne modifie pas le couple précédent.

A la fin le programme affiche le couple optimal et également une coubre qui trace l'Espérance au carré par rapport
à l'écart entre C_in_prime et C_out_prime.

                                                                            


"""

""" Méthode Optimisation ==>> mode unidirectionnel  """



import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from scipy import sparse

## Penser à faire un graph pour vérifier l'intensité du bruit
## ++ grand nombre 


#On simule un graph G de matrice d'adjacence A à l'aide stochastic bloc model
# V=[v_1,...v_n] : ensemble des points
# C : C[i] correspond au cluster du point V[i] 
#c_in et c_out sont liées aux probabilités du stochastic bloc model  

def stocha(V,C,c_in,c_out,c_in_prime,c_out_prime,p1,p2) : 
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
    

def g(A,p1,p2) : 
    
    U = clusteringspectral(A,2) 
    C1=PrintCluster(C)[0]
    C2=PrintCluster(C)[1]
    B1=PrintCluster(U)[0]
    B2=PrintCluster(U)[1]
    M = Confusionmatrix(C1,C2,B1,B2)
    i2=0
    h1=0
    ClusterassocieC1 = PrintCluster(U)[np.argmin(M[0,:])]
    ClusterassocieC2 = PrintCluster(U)[np.argmax(M[0,:])]
    
    for k in list1:
        if k in ClusterassocieC1:
            h1=h1+1
            
    for k in list2:
        if k in ClusterassocieC2:
            i2=i2+1

    if (h1==p1)&(i2==p2):
        return 0
    else:
        #print("ClusteAssocie")
        #print(ClusterassocieC1[:3])
        #print(ClusterassocieC2[:3])
        return 1 

    
#######################################################
######## Données du problème ##########################






n=100 #nombre de points
k=2  #nombre de cluster
M = 5000
V=np.arange(n)  
C=np.zeros(n)    
nprime=n/2
C[nprime:]=np.ones(n-nprime)
###############
c_in=35.0
c_out=17.0
p_in=c_in/n
p_out=c_out/n


################ Creation de la liste d indices / ou sommets
p1=2
p2=0
list1 = []
list2 = []
for h in range(p1):
    list1.append(h)
n0 = int(nprime)
n1 = int(nprime+p2)
for h in range(n0,n1):
    list2.append(h)





print("##################  Rechercher des paramètres optimaux    ####################")
print("Données du problème :")
print("Nombre de simulation",M)
print("C_in",c_in)
print("C_out",c_out)
print("##################  Calcul ################ ")


### Valeur Arbitraire idée : Ecart_min = + infini
# Ecart_min[0] = la valeur de l'Ecart_type
# Ecart_min[1] = stocke le C_in_prime qui a donné Ecart_min[0]
# Ecart_min[2] = stocke le C_out_prime qui a donné Ecart_min[0]
Min = [20,0,0] 


#################
#################   Plage de test pour les coefficients C_in_prime C_out_prime
#################
step = 0.1 
Nbr_steps = 20

C_in_prime_max = c_in
C_in_prime_min = C_in_prime_max - step*Nbr_steps ### Possibilité de mettre directement c_in


C_out_prime_min = c_out 
C_out_prime_max = C_out_prime_min + step*Nbr_steps 

Err_tab = []
Abscisse_tab = [] 

print(C_out_prime_min)
print(C_out_prime_max)

List_c_in = np.arange(C_in_prime_min,C_in_prime_max,step)
List_c_out = np.arange(C_out_prime_min,C_out_prime_max,step)

print("List_c_in",List_c_in)
print("List_c_out",List_c_out)


for h in range(Nbr_steps):
    c_in_prime = List_c_in[Nbr_steps-h-1]
    c_out_prime = List_c_out[h]
    #Debug
    print("c_in_prime",c_in_prime)
    print("c_out_prime",c_out_prime)
    p_in_prime=c_in_prime/n
    p_out_prime=c_out_prime/n

    ### Coeffiicent multiplicateur  ==>> modification pour chaque couple ≠
    S_in = (1-p_in)/(1-p_in_prime)
    S_in_simple = p_in/p_in_prime
    S_out = (1-p_out)/(1-p_out_prime)
    S_out_simple = (p_out)/(p_out_prime)


    Echantillon_carre = []
    ### Pour un couple de (C_in_prime,C_out_prime)
    for i in range(M):
        A=stocha(V,C,c_in,c_out,c_in_prime,c_out_prime,p1,p2)
        Z_in = A[1]
        Z_out = A[2]
        Echantillon_carre.append(g(A[0],p1,p2)*((S_in**Z_in[0])*(S_in_simple**Z_in[1])*(S_out**Z_out[0])*(S_out_simple**Z_out[1]))**2)
    

    ## Valeur intéressante
    # But : récupérer le couple (C_in_prime,C_out_prime) qui minimise EsperanceCarre
    #
    EsperanceCarre = np.mean(Echantillon_carre)
    print(Echantillon_carre)
    print("EsperanceCarre",EsperanceCarre)
    print("boolean", EsperanceCarre < Min[0])
    
    Err_tab.append(EsperanceCarre)
    Abscisse_tab.append(c_in_prime-c_out_prime )
    
    #Garde de fou
    if EsperanceCarre == 0 : 
        print("Simulation non probante : Evenement trop rare Ecart_type=0")
    else :
        if EsperanceCarre < Min[0] :
            Ecart_min =  [EsperanceCarre,c_in_prime,c_out_prime]
            print(Ecart_min)
###

# Faire un graphr afin de voir le résultat est intéressant 

#####      Fin de la boucle     ######
print("###########     Résultat    ################")
print("Couple optimal :  ")
print("Ecart_min ",Ecart_min[0] ,"C_in_prime : ", Ecart_min[1],"C_out_prime : ", Ecart_min[2])

print("Err_tab",Err_tab)
print("Abscisse_tab",Abscisse_tab)

plt.xlabel('Cin-Cout')
plt.ylabel('Demi-largeur')
plt.title('Cherche des paramètres cin_prime et cout_prime optimaux')
plt.plot(Abscisse_tab,Err_tab)
plt.show()


    
    










        
     






















    
    





