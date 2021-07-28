# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:55:18 2019

@author: fede
"""

#The N modes gaussian states are represented as Z, a NxN complex valued adjacency matrix as described in https://arxiv.org/abs/1007.0725
#or as CM, a 2Nx2N covariance matrix.
#The notation for the quadratures is to stack the position vectors of operators q on top of the momentum vectors of operators p inside a vector x. 
#Hence the first N entries of the covariance matrix will refer to position and the others to momentum.


import matplotlib.pyplot as plt 
from matplotlib import cm
import random
import networkx as nx

from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
#import scipy as sp
#import matInv

#generates the covariance matrix (divided by h/2pi) starting from the complex adjacency matrix
def _generateCovM(Zz):
    U=np.imag(Zz)
    V=np.real(Zz)
    invU = np.linalg.inv(U)
    CM=0.5*np.block([[invU, invU @ V],[V @ invU, U+V @ invU@V]])
    
    return CM 

#generates the the complex adjacency matrix starting from the covariance matrix 
def _generateZ(CM):
    dim=int(len(CM)/2)
    Ui=np.empty([dim,dim])
    UiV=np.empty([dim,dim])
    for i in range(dim):
        for j in range(dim):
            Ui[i,j]=CM[i,j]
            UiV[i,j]=CM[i+dim,j]
    U=np.linalg.inv(Ui)/2
    V=U@UiV*2
    return V+1j*U
    

#builds the symplectic form matrix
def _buildSymM(dim):
    return np.block([[np.zeros((dim,dim)),np.identity(dim)], [-np.identity(dim),np.zeros((dim,dim))]])

#Returns the purity of the state from its covariance matrix
def _Purity(CM):
    det=np.linalg.det(CM)
    Purity=1/((2**(len(CM)/2))*np.sqrt(det))
    return Purity

#Checks if a matrix is positive defined
def _MisPositive(MM):
    lambdas=np.linalg.eig(MM)[0]
    #thr=1e14
    thr=np.abs(sum(lambdas))/len(lambdas)*1e-5
    realCheck=np.amax(np.abs(np.imag(lambdas)))>thr
    if realCheck == True:
        print("The matrix has complex eigenvalues: \n", lambdas)
    PosCheck=np.amin(np.real(lambdas))<-thr
    if PosCheck == True :
        print("The matrix has negative eigenvalues!\n", lambdas)
    return not (PosCheck or realCheck) 

#Checks the gaussianity of a covariance matrix
def _checkGaussianity(CM):
    return _MisPositive(CM+0.5j*_buildSymM(int(len(CM)/2)))

#Sum of the variance of the nullifiers
def _nullifiers(Zz):
    return 0.5*np.trace(np.imag(Zz))

#Logarithmic negativity of entanglement, for two modes only
def negativity(Zz):
    if len(Zz)==2:
        CM=_generateCovM(Zz)
        sym=_buildSymM(len(Zz))
    elif len(Zz)==4:
        CM=Zz
        sym=_buildSymM(int(len(Zz)/2))
    gamma=np.diag([1,1,-1,1])
    CMpt=gamma@CM@gamma
    lamb=np.min(np.abs(np.linalg.eig(1j*sym@(CMpt))[0]))
    return np.max([0,-np.log(2*lamb)])
#    return np.min(eig)


#Energy of the state
def DeltaE(Zz):
    CM=_generateCovM(Zz)
    return 0.5*(np.trace(CM)-len(Zz))

#Compute Wigner function
def _Wigner(sigma):
    Vrange=2
    steps=100
    qq=np.linspace(-Vrange,Vrange,steps)
    pp=np.copy(qq)
    det=np.linalg.det(sigma)
    WW=np.zeros([steps,steps])
    for i in range(steps):
        for j in range(steps):
            xx=np.array([qq[i],pp[j]])
            esp=xx@(np.linalg.inv(sigma))@(np.transpose(xx))
            WW[i,j]=(2*np.pi)**(-int(len(sigma)/2))*det**(-0.5)*np.exp(-0.5*esp)
    return WW

#Compute the reduced state of 'node'
def _partial(Zz,node):
    CM=_generateCovM(Zz)
    size=int(len(CM)/2)
    redCM=np.zeros([2,2])
    redCM[0,0]=CM[node, node]
    redCM[0,1]=CM[node,node+size]
    redCM[1,0]=redCM[0,1]
    redCM[1,1]=CM[node+size,node+size]
    return redCM

#Reduced state of node1 and node2
def _2partial(Zz,node1,node2):
    CM=_generateCovM(Zz)
    size=int(len(CM)/2)
    redCM=np.zeros([4,4])
    redCM[0,0]=CM[node1, node1]
    redCM[1,1]=CM[node2, node2]
    redCM[2,2]=CM[node1+size, node1+size]
    redCM[3,3]=CM[node2+size, node2+size]
    redCM[0,1]=CM[node1,node1+size]
    redCM[0,2]=CM[node1,node2]
    redCM[0,3]=CM[node1,node2+size]
    redCM[1,2]=CM[node2,node1+size]
    redCM[1,3]=CM[node2,node2+size]
    redCM[2,3]=CM[node1+size,node2+size]
    redCM[1,0]=redCM[0,1]
    redCM[2,0]=redCM[0,2]
    redCM[3,0]=redCM[0,3]
    redCM[2,1]=redCM[1,2]
    redCM[3,1]=redCM[1,3]
    redCM[3,2]=redCM[2,3]
    return redCM

#Squeezing cost of the state  
def squeezeCost(Zz):
    dim=len(Zz)
    SUM=0
    eig=np.linalg.eig(2*_generateCovM(Zz))[0]
    eig=-np.sort(-eig)
    for x in range(dim):
        SUM+=10*np.log(eig[x])/np.log(10)
    return np.real(SUM)

#Number of modes in the gaussian state
def NModes(Zz):
    SUM=0
    eig=np.linalg.eig(2*_generateCovM(Zz))[0]
    eig=-np.sort(-eig)
    sq=squeezeCost(Zz)
    x=0
    while SUM<0.99*sq:
        SUM+=10*np.log(eig[x])/np.log(10)
        x+=1
    return x

#Squeezing spectrum of the state
def histoSqueeze(Zz):
    eig=np.linalg.eig(2*_generateCovM(Zz))[0]
    eig=np.sort(eig)
    eig=eig[np.arange(len(Zz))]    
    return np.abs(10*np.log(eig)/np.log(10))
#    return eig

#Adjacency matrix of the 
def adjMat(Zz):
    return np.abs(Zz-np.diag(np.diag(Zz)))
    
#Spectrum of adj matrix
def histoAdj(Aa):
    eig=np.linalg.eig(Aa)[0]
    ki=np.sqrt(1+eig**2)
    ki=-np.sort(-ki)
    return ki


    
#Plot wigner function
def PlotWigner(CM):
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    
    # Make data.
    X = np.linspace(-Vrange, Vrange, steps)
    Y = np.linspace(-Vrange, Vrange, steps)
    X, Y = np.meshgrid(X, Y)
    W=_Wigner(CM)
    
    # Plot the surface.
    surf = ax.plot_surface(X, Y, W, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
    
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
    return 

#Count the number of edges in the state Z
def connections(Zz):
    return np.count_nonzero(Zz-np.diag(np.diag(Zz)))/2

#################Gaussian transformations on Z#################

#Multimode squeezing
def MultiSqueeze(Zz, ss): #s must be an array not a list
    SS=np.identity(len(Zz))*np.power(10.,-ss/10)
    return SS.dot(Zz)
#Two modes beamsplitter between modes node1 and node2 with angle phi
def BeamSplitter(Zz, node1,node2,phi):
    newZz=np.copy(Zz)
    newZz[node1,:]=Zz[node1,:]*np.cos(phi)-Zz[node2,:]*np.sin(phi)
    newZz[:,node1]=newZz[node1,:]
    newZz[node2,:]=Zz[node2,:]*np.cos(phi)-Zz[node1,:]*np.sin(phi)
    newZz[:,node2]=newZz[node2,:]
    newZz[node1,node2]=Zz[node1,node2]*np.cos(2*phi)+np.sin(2*phi)/2*(Zz[node1,node1]-Zz[node2,node2])
    newZz[node2,node1]=newZz[node1,node2]
    newZz[node1,node1]=Zz[node1,node1]*np.cos(phi)**2+Zz[node2,node2]*np.sin(phi)**2+Zz[node1,node2]*np.sin(2*phi)
    newZz[node2,node2]=Zz[node1,node1]*np.sin(phi)**2+Zz[node2,node2]*np.cos(phi)**2+Zz[node1,node2]*np.sin(2*phi)
    return newZz

#CZ gate between node1 and node2 with intensity g
def CZgate(Zz, g, node1,node2):
    Zz[node1,node2]=Zz[node1,node2]+g
    Zz[node2,node1]=Zz[node1,node2]
    return Zz

#Position quadrature measurement
def MeasureQ(Zz, node):
    Zz=np.delete(Zz, (node), axis=0)
    Zz=np.delete(Zz, (node), axis=1)
    return Zz

#Quadrature phase rotation of angle phi
def PhaseShift(ZZ, node, phi):
    ZZ=np.matrix(ZZ)
    newZZ=ZZ-np.sin(phi)*np.matmul(ZZ[:,node],ZZ[node,:])/(ZZ[node,node]*np.sin(phi)+np.cos(phi))
    newZZ[node,:]=ZZ[node,:]/(ZZ[node,node]*np.sin(phi)+np.cos(phi))
    newZZ[:,node]=ZZ[:,node]/(ZZ[node,node]*np.sin(phi)+np.cos(phi))
    newZZ[node,node]=(-np.sin(phi)+ZZ[node,node]*np.cos(phi))/(ZZ[node,node]*np.sin(phi)+np.cos(phi))
    
    return np.array(newZZ)
    
#Momentum quadrature measurement
def MeasureP(Zz, node):
    return MeasureQ(PhaseShift(Zz, node, np.pi/2), node)


#################State Generation#################
    
#Generates the state of N-dimensional vacuum state
def VacuumState(Nn):
    return 1j*np.identity(Nn, dtype=np.cdouble)

#Generates a linear graph state
def LinGraph(Nn, gg, ss):
    Zlin=MultiSqueeze(VacuumState(Nn),ss)
    for i in range(Nn-1):
        Zlin=CZgate(Zlin,gg,i,i+1)
    return Zlin
        
#Generates a circular graph state
def circGraph(Nn,gg, ss):
    zz=LinGraph(Nn,gg,ss)
    return CZgate(zz,gg,0, Nn-1)

#Generates an EPR pair
def EPR(sss):
    sqZ=MultiSqueeze(VacuumState(2),np.array([sss,-sss]))
    return BeamSplitter(sqZ,0,1,np.pi/4)

#Generates a star graph state
def starGraph(Nn,gg,ss):
    Zstar=MultiSqueeze(VacuumState(Nn),ss)
    for i in range(Nn-1):
        Zstar=CZgate(Zstar,gg,0,i+1)
    return Zstar

#Generates a diamond graph state
def Diamond(Nn,gg,ss):
    state=MultiSqueeze(VacuumState(Nn),ss)
    for i in range(1,Nn-1):
        state=CZgate(state,gg,0,i)
        state=CZgate(state,gg,Nn-1,i)
    return state

#def clusterState(W,H,gg,ss):
#    state=MultiSqueeze(VacuumState(W*H),ss)
#    for j in range(W):
#        for i in range(H):
#            if i==H-1 and j!=W-1:
#                state=CZgate(state,gg,i+j*W,i+(j+1)*W)                
#            elif j==W-1 and i!=H-1:
#                state=CZgate(state,gg,i+j*W,i+j*W+1)
#            elif i!=H-1 or j!=W-1:
#                state=CZgate(state,gg,i+j*W,i+j*W+1)
#                state=CZgate(state,gg,i+j*W,i+(j+1)*W)
#    return state

#generates a rectangular cluster state
def clusterState(rows,cols,gg,ss):
    state=MultiSqueeze(VacuumState(rows*cols),ss)    
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            # Two inner diagonals
            if c > 0: CZgate(state,gg,i-1,i)
            # Two outer diagonals
            if r > 0: CZgate(state,gg,i-cols,i) 
    return state

def triclusterState(rows,cols,gg,ss):
    state=MultiSqueeze(VacuumState(rows*cols),ss)    
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            # Two inner diagonals
            if c > 0: CZgate(state,gg,i-1,i)
            # Two outer diagonals
            if r > 0: CZgate(state,gg,i-cols,i) 
            if r>0 and c<cols-1: CZgate(state,gg,i-cols+1,i) 
    return state

def triclusterState2(rows,cols,gg,ss):
    state=MultiSqueeze(VacuumState(rows*cols),ss)    
    for r in range(rows):
        for c in range(cols):
            i = r*cols + c
            # Two inner diagonals
            if c > 0: CZgate(state,gg,i-1,i)
            # Two outer diagonals
            if r > 0: CZgate(state,gg,i-cols,i) 
            if r>0 and c>0: CZgate(state,gg,i-cols-1,i) 
    return state

def triangleLattice(rows,cols,gg,ss):
    state=MultiSqueeze(VacuumState(rows*cols),ss)
    graph=nx.generators.lattice.triangular_lattice_graph(rows,cols)
    Aa=nx.adjacency_matrix(graph)
    for i in range(rows):
        for j in range(cols):
            state=CZgate(state,gg*Aa[i,j+i],i,j+i)
    return state

def NXgraphDiamond(Nn):
    G = nx.Graph()
    G.add_node(0)
    G.add_node(Nn-1)
    for i in range(1,Nn-1):
        G.add_node(i)
        G.add_edge(i,0)
        G.add_edge(i,Nn-1)
    return G

def DiamondChain(R,C,gg,ss):
    n=R*C+2
    A=np.zeros([n,n])
    for i in range(R):
        A[0,C*i+1]=1
        A[C*i+1,0]=1
        A[C*i+C,n-1]=1
        A[n-1,C*i+C]=1
        for i in range(R*C):
            if (i+1)%C!=0:
                A[i+1, i+2]=1
                A[i+2,i+1 ]=1
    state=MultiSqueeze(VacuumState(n),ss)
    for i in range(n):
        for j in range(n-i):
            state=CZgate(state,gg*A[i,j+i],i,j+i)
    return state

#def extenDiamond(Nn,Cc,gg,ss):
#    state=MultiSqueeze(VacuumState(2+Nn*Cc),ss)
#    for i in range(1,Nn-1):
#        state=CZgate(state,gg,0,i)
#        state=CZgate(state,gg,Nn-1,i)
#    return state

def Lin2Cliq(Nn,Kk,gg,ss):
    state=MultiSqueeze(VacuumState(Kk*Nn+Nn-Kk),ss)
    for i in range(Nn-1):
        for j in range(Kk):
            state=CZgate(state,gg,i*(Kk+1),(Kk+1)*i+j+1)
            state=CZgate(state,gg,i*(Kk+1)+Kk+1,(Kk+1)*i+j+1)
    return state
    
    


def multiPathBS(Nn,ss):
    squeezing=np.zeros(Nn+2)
    squeezing[0]=ss
    squeezing[Nn+1]=-ss
    state=MultiSqueeze(VacuumState(Nn+2),squeezing)
    for i in range(1,Nn+1):
        state=BeamSplitter(state,0,i,np.pi/4)
        state=BeamSplitter(state,Nn+1,i,np.pi/4)
    return state

def fullGraph(Nn,gg,ss):
    state=MultiSqueeze(VacuumState(Nn),ss)
    for i in range(Nn-1):
        for j in range(Nn-i-1):
            state=CZgate(state,gg,i,i+j+1)
    return state   

#Complex network graph states

def barabasi_albert(Nn,Mm,seed,gg,ss):
    graph=nx.adjacency_matrix(nx.barabasi_albert_graph(Nn,Mm,seed))
    state=MultiSqueeze(VacuumState(Nn),ss)
    for i in range(Nn):
        for j in range(Nn-i):
            state=CZgate(state,gg*graph[i,j+i],i,j+i)
    return state

def watts_strogatz(Nn,Kk,p,seed,gg,ss):
    graph=nx.adjacency_matrix(nx.watts_strogatz_graph(Nn,Kk,p,seed))
    state=MultiSqueeze(VacuumState(Nn),ss)
    for i in range(Nn):
        for j in range(Nn-i):
            state=CZgate(state,gg*graph[i,j+i],i,j+i)
    return state

def erdos_renyi(Nn,beta,seed,gg,ss):
    graph=nx.adjacency_matrix(nx.erdos_renyi_graph(Nn,beta))
    state=MultiSqueeze(VacuumState(Nn),ss)
    for i in range(Nn):
        for j in range(Nn-i):
            state=CZgate(state,gg*graph[i,j+i],i,j+i)
    return state

#General graph state given a nx graph
def graphState(NXgraph,gg,ss):
    Nn=len(NXgraph)
    Aa=nx.adjacency_matrix(NXgraph)
    state=MultiSqueeze(VacuumState(Nn),ss)
    for i in range(Nn):
        for j in range(Nn-i):
            state=CZgate(state,gg*Aa[i,j+i],i,j+i)
    return state
