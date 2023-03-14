# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:40:53 2023

@author: gabri
"""

import numpy as np 
from sklearn.cluster import KMeans
from numba import jit, prange
import random as rn
from copy import deepcopy

class RBF():
    def __init__(self, input_length, hidden_length, output_lenght):
        
        self.input_length= input_length
        self.hidden_lenght= hidden_length
        self.output_lenght= output_lenght
        self.gammas= np.ones(hidden_length)
    
    def acha_centros(self, previsores, max_interacoes=1000):
        km = KMeans(n_clusters= self.hidden_lenght, max_iter=max_interacoes, verbose=0)
        km.fit(previsores)
        self.centros= km.cluster_centers_
        max_dist= self.max_dist_Kmeans(self.centros)
        self.gammas*= max_dist/(2*self.hidden_lenght)**(1/2)
        
    def train(self, previsores, real):        
        self.ys_train= real
        self.previsores_train= previsores
        self.previsores_train_lenght= len(previsores)
        self.acha_centros(previsores)
        
        
        
        self.matriz_dist= calcula_dist(centros= self.centros, n_centros= self.hidden_lenght, 
                                       base= self.previsores_train, base_lenght= self.previsores_train_lenght)
       
        matriz_fi= aplica_gaussiana(dist= self.matriz_dist, fi_lenght= self.previsores_train_lenght, 
                                    gammas= self.gammas, gammas_lengt= self.hidden_lenght)
        
        self.calcula_pesos(matriz_fi, self.ys_train)




    def forward(self, previsores, gammas=None):  
        
        self.tam_previsores= len(previsores)
        self.previsores= previsores 
        
        matriz_dist= calcula_dist(centros= self.centros, n_centros= self.hidden_lenght, 
                                       base= self.previsores, base_lenght= self.tam_previsores)                                   
        
       
        if gammas is None:
            
            matriz_fi= aplica_gaussiana(dist= matriz_dist, fi_lenght= self.tam_previsores, gammas= self.gammas, 
                                    gammas_lengt= self.hidden_lenght)           
            
        else:
            
             matriz_fi= aplica_gaussiana(fi= matriz_dist, fi_lenght= self.tam_previsores, gammas= gammas, 
                                     gammas_lengt= self.hidden_lenght)
         
        resultado= self.calcula_resultados(matriz_fi)
        
        return resultado
    
   
    def calcula_pesos(self,fi, y_s):    
        pseudo_inv= np.linalg.pinv(fi)
        self.pesos=  pseudo_inv @ y_s
    
    def calcula_resultados(self,fi):   
        return fi @ self.pesos
    
    def max_dist_Kmeans(self,x):
        max_dist=0
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                dist=  (x[i]-x[j]) @ (x[i]-x[j]).T                
                if max_dist < dist:
                    max_dist= dist
        return int(max_dist)
    
    
    def test_gammas_ed(self, gammas):
        
        
        
        matriz_fi= aplica_gaussiana(dist= self.matriz_dist, fi_lenght= self.previsores_train_lenght
                                         , gammas= gammas, gammas_lengt= self.hidden_lenght)
        
       
        
        self.calcula_pesos(matriz_fi, self.ys_train)
        
        return  self.calcula_resultados(matriz_fi)

   
   
    
    def train_ED(self, tam_pop, largura_max, previsores_train, y_train,
            n_geracoes, f_escala=0.2, prop_mut=0.3):
        
        self.matriz_dist= calcula_dist(centros= self.centros, n_centros= self.hidden_lenght, 
                                       base= previsores_train, base_lenght= len(previsores_train))
        
        
        self.gammas=  self.E_D(tam_pop, largura_max, previsores_train, y_train,
                n_geracoes, f_escala=0.2, prop_mut=0.3)
        
        matriz_fi= aplica_gaussiana(dist= self.matriz_dist, fi_lenght= self.previsores_train_lenght, 
                                    gammas= self.gammas, gammas_lengt= self.hidden_lenght)
        
        
        
        self.calcula_pesos(matriz_fi, self.ys_train)
       
        
    
    def E_D(self, tam_pop, largura_max, previsores_train, y_train,
            n_geracoes, f_escala=0.2, prop_mut=0.3):
        
        # gera população inicial
        gammas= np.random.normal(loc=self.gammas[0] , scale= 0.01, size=(tam_pop, self.hidden_lenght)).round(2)
        fitnes_pop= []
        
       
        
        # calcula erro da pop inicial
        for i in range(tam_pop):
            
            result= self.test_gammas_ed(gammas[i])           
            erro= self.mean_absolute_error(real= y_train, previsto=result)            
            fitnes_pop.append(erro)
        
       
        
        for i in range(n_geracoes):  
            new_gammas= []
            fitnes_new_popy=[]
           
            print ('\r',f'Geração: {i}', end='', flush=True)
            for j in range(tam_pop):

                #mutacao
                indices= rn.sample(range(0,tam_pop), 3)
                u_g= gammas[indices[0]]+ f_escala*(gammas[indices[1]]-gammas[indices[2]])

                #cruzamento
                for k in range(self.hidden_lenght):
                    if rn.random()>prop_mut:                    
                        u_g[k]= gammas[j][k] 

                result= self.test_gammas_ed(u_g)                
                fitnes_u_g= self.mean_absolute_error(real= result, previsto= y_train)
                fitnes_x= fitnes_pop[j]
                #selecao
                
                if fitnes_u_g<fitnes_x:
                    
                    fitnes_new_popy.append( fitnes_u_g)
                    new_gammas.append( u_g)
                else:               
                    fitnes_new_popy.append( fitnes_x)
                    new_gammas.append( gammas[j])
             
                
        
            gammas= deepcopy(new_gammas)
            fitnes_pop= deepcopy(fitnes_new_popy)
            menor= float ('inf')
            
                
           
            
                
       
        for i in range(tam_pop):        
            if menor > fitnes_pop[i]:
                menor= fitnes_pop[i]
                indice=i
        
       
        return gammas[indice]
    

    def mean_absolute_error(self, real, previsto):    
        return abs(real-previsto).sum()/len(real)



@jit(nopython=True, parallel=True)
def calcula_dist(centros, n_centros, base, base_lenght):    
    dist= np.ones((base_lenght, n_centros+1))    
    
    # entrada - centros
    for i in prange(base_lenght):      
        for j in prange(n_centros):    
            d= centros[j]- base[i]
            dist[i,j]=  np.dot( d ,  d.T)   
    return dist

@jit(nopython=True, parallel=True)
def aplica_gaussiana(dist, fi_lenght, gammas, gammas_lengt):
    
    fi= np.ones((fi_lenght, gammas_lengt+1))
    for i in prange(fi_lenght):
        for j in prange(gammas_lengt):
            fi[i,j]= 2.71828**(-1/(2*gammas[j]**2)* dist[i,j])
    
    return fi
            
    

    


    
    
if __name__ == '__main__':
    xor=np.array([[1,1,1],[0,0,1],[0,1,0],[1,0,0]], dtype= np.float64)
    
    rbf=  RBF(input_length= 2, hidden_length= 4, output_lenght= 1)
    
    rbf.train(previsores= xor[:,:2],real= xor[:,2])
    print('resultado train normal sem passar gammas',rbf.forward(xor[:,:2]))
    
    rbf.train_ED(tam_pop=3, largura_max=1, previsores_train=xor[:,:2],
                 y_train=xor[:,2] , n_geracoes=1)

    print('resultado train ed',rbf.forward(xor[:,:2]))