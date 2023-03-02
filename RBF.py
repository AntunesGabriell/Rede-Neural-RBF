# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 15:40:53 2023

@author: gabri
"""

import numpy as np 
from sklearn.cluster import KMeans

class RBF():
    def __init__(self, input_length, hidden_length, output_lenght):
        
        self.input_length= input_length
        self.hidden_lenght= hidden_length
        self.output_lenght= output_lenght
        self.gammas= np.ones(hidden_length)
    
    def acha_centros(self, previsores, max_interacoes=100):
        km = KMeans(n_clusters= self.hidden_lenght, max_iter=max_interacoes, verbose=0)
        km.fit(previsores)
        self.centros= km.cluster_centers_
        max_dist= self.max_dist_Kmeans(self.centros)
        self.gammas*= max_dist/(2*self.hidden_lenght)**(1/2)
        
    def train(self, previsores, real, epocas):        
        tam_base= len(previsores)
        self.acha_centros(previsores)
        matriz_fi= self.calcula_hidden(previsores, tam_base)
        
        self.calcula_pesos(matriz_fi, real)


    def forward(self, previsores):     
        tam_base= len(previsores)
        matriz_fi= self.calcula_hidden(previsores, tam_base)              
        resultado= self.calcula_resultados(matriz_fi)
        
        return resultado
    
    def calcula_hidden(self,previsores, tam_base):
        fi= np.ones((tam_base, self.hidden_lenght+1))
        # entrada - centros
        for i in range(tam_base):      
            for j in range(self.input_length):     
                fi[i,j]=  (self.centros[j]- previsores[i]) @ (self.centros[j]- previsores[i] ).T
                fi[i,j]= 2.71828**(-1/(2*self.gammas[j]**2)* fi[i,j])  
        
        return fi               
    
    
    def calcula_pesos(self,fi, y_s):    
        pseudo_inv= np.linalg.pinv(fi)
        self.pesos=  pseudo_inv @ y_s
    
    def calcula_resultados(self,fi):   
        print('shape fi', np.shape(fi))
        print('shape pesos', np.shape(self.pesos))
        return fi @ self.pesos
    
    def max_dist_Kmeans(self,x):
        max_dist=0
        for i in range(len(x)):
            for j in range(i+1, len(x)):
                dist=  (x[i]-x[j]) @ (x[i]-x[j]).T                
                if max_dist < dist:
                    max_dist= dist
        return int(max_dist)
    
    

def xor():
    xor= np.array([[1,1,1],[0,1,0],[0,0,1],[1,0,0]])

    rbf=  RBF(input_length= 2, hidden_length= 4, output_lenght= 2)
    rbf.train(previsores= xor[:,:2],real= xor[:,2],epocas=10)
    rbf.centros= np.array([[1,1],[0,0]])

    print('resultado',rbf.forward(xor[:,:2]))

if __name__ == '__main__':
    xor()





