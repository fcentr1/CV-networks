#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 28 15:11:40 2021

@author: fede
"""





from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np 
import networkx as nx
import random 
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from networkx.drawing.nx_agraph import graphviz_layout
import time
import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import csv



#generates the covariance matrix (divided by h/2pi)
def _generateCovM(Zz):
    U=np.imag(Zz)
    V=np.real(Zz)
    invU = np.linalg.inv(U)
    CM=0.5*np.block([[invU, invU @ V],[V @ invU, U+V @ invU@V]])
    return CM 


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



def niggativity(Zz):
    if len(Zz)==2:
        CM=_generateCovM(Zz)
    elif len(Zz)==4:
        CM=Zz
    I1=CM[0,0]*CM[2,2]-CM[0,2]**2
    I2=CM[1,1]*CM[3,3]-CM[1,3]**2
    I3=CM[0,1]*CM[2,3]-CM[1,2]*CM[0,3]
    I4=np.linalg.det(CM)
    DELTA=I1+I2-2*I3
    lamb=np.sqrt(0.5*(DELTA-np.sqrt(DELTA**2-4*I4)))
    return np.max([0,-np.log(2*lamb)])
#    return lamb

def Fcoh(Zz):
    if len(Zz)==2:
        CM=_generateCovM(Zz)
        sym=_buildSymM(len(Zz))
    elif len(Zz)==4:
        CM=Zz
        sym=_buildSymM(int(len(Zz)/2))
    gamma=np.diag([1,1,-1,1])
    CMpt=gamma@CM@gamma
    eig=np.min(np.abs(np.linalg.eig(2j*sym@(CMpt))[0]))     
    return 1/(1+eig)




def Fcoh3(Zz):
    if len(Zz)==2:
        CM=_generateCovM(Zz)
    elif len(Zz)==4:
        CM=Zz    
    dx2=(CM[0,0]+CM[1,1]-2*CM[0,1])
    dp2=(CM[2,2]+CM[3,3]+2*CM[2,3])
    return 1/np.sqrt((1+dx2)*(1+dp2))

def DeltaE(Zz):
    CM=_generateCovM(Zz)
    return 0.5*(np.trace(CM)-len(Zz))

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

def _partial(Zz,node):
    CM=_generateCovM(Zz)
    size=int(len(CM)/2)
    redCM=np.zeros([2,2])
    redCM[0,0]=CM[node, node]
    redCM[0,1]=CM[node,node+size]
    redCM[1,0]=redCM[0,1]
    redCM[1,1]=CM[node+size,node+size]
    return redCM

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

  
def squeezeCost(Zz):
    dim=len(Zz)
    SUM=0
    eig=np.linalg.eig(2*_generateCovM(Zz))[0]
    eig=-np.sort(-eig)
    for x in range(dim):
        SUM+=5*np.log(eig[x])/np.log(10)
    return np.real(SUM)

def NModes(Zz):
    SUM=0
    eig=np.linalg.eig(2*_generateCovM(Zz))[0]
    eig=-np.sort(-eig)
    sq=squeezeCost(Zz)
    x=0
    while SUM<0.99*sq:
        SUM+=5*np.log(eig[x])/np.log(10)
        x+=1
    return x

def histoSqueeze(Zz):
    eig=np.linalg.eig(2*_generateCovM(Zz))[0]
    eig=np.sort(eig)
    eig=eig[np.arange(len(Zz))]    
    return np.abs(5*np.log(eig)/np.log(10))
#    return eig

def adjMat(Zz):
    return np.abs(Zz-np.diag(np.diag(Zz)))
    

def histoAdj(Aa):
    eig=np.linalg.eig(Aa)[0]
    ki=np.sqrt(1+eig**2)
    ki=-np.sort(-ki)
    return ki

def graphEnt(Z):
    VV=np.real(Z[0,1])
    Ra=np.imag(Z[0,0])
    Rb=np.imag(Z[1,1])
    dsig=0.5+VV**2/(Ra*Rb)
    EE=-np.log(np.sqrt(0.5*(dsig-np.sqrt(dsig**2-1/4))))/np.log(2)
    return np.max([0,EE])

    



def connections(Zz):
    return np.count_nonzero(Zz-np.diag(np.diag(Zz)))/2

#################Gaussian transformations on Z#################
    
def MultiSqueeze(Zz, ss): #s must be an array not a list
    SS=np.identity(len(Zz))*np.power(10.,-2*ss)
    return SS.dot(Zz)

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

def CZgate(Zz, g, node1,node2):
    Zz[node1,node2]=Zz[node1,node2]+g
    Zz[node2,node1]=Zz[node1,node2]
    return Zz

def MeasureQ(Zz, node):
    Zz=np.delete(Zz, (node), axis=0)
    Zz=np.delete(Zz, (node), axis=1)
    return Zz

def PhaseShift(ZZ, node, phi):
    ZZ=np.matrix(ZZ)
    newZZ=ZZ-np.sin(phi)*np.matmul(ZZ[:,node],ZZ[node,:])/(ZZ[node,node]*np.sin(phi)+np.cos(phi))
    newZZ[node,:]=ZZ[node,:]/(ZZ[node,node]*np.sin(phi)+np.cos(phi))
    newZZ[:,node]=ZZ[:,node]/(ZZ[node,node]*np.sin(phi)+np.cos(phi))
    newZZ[node,node]=(-np.sin(phi)+ZZ[node,node]*np.cos(phi))/(ZZ[node,node]*np.sin(phi)+np.cos(phi))
    
    return np.array(newZZ)
    
def MeasureP(Zz, node):
    return MeasureQ(PhaseShift(Zz, node, np.pi/2), node)


#################State Generation#################
    
#Generates the state of N-dimensional vacuum state
def VacuumState(Nn):
    return 1j*np.identity(Nn, dtype=np.cdouble)

#Generates a linear cluster state

def LinGraph(Nn, gg, ss):
    Zlin=MultiSqueeze(VacuumState(Nn),ss)
    for i in range(Nn-1):
        Zlin=CZgate(Zlin,gg,i,i+1)
    return Zlin
        
def circGraph(Nn,gg, ss):
    zz=LinGraph(Nn,gg,ss)
    return CZgate(zz,gg,0, Nn-1)

def EPR(sss):
    sqZ=MultiSqueeze(VacuumState(2),np.array([sss,-sss]))
    return BeamSplitter(sqZ,0,1,np.pi/4)

def starGraph(Nn,gg,ss):
    Zstar=MultiSqueeze(VacuumState(Nn),ss)
    for i in range(Nn-1):
        Zstar=CZgate(Zstar,gg,0,i+1)
    return Zstar

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

def graphState(NXgraph,gg,ss):
    Nn=len(NXgraph)
    Aa=nx.adjacency_matrix(NXgraph)
    state=MultiSqueeze(VacuumState(Nn),ss)
    for i in range(Nn):
        for j in range(Nn-i):
            state=CZgate(state,gg*Aa[i,j+i],i,j+i)
    return state

def MeasAllQ(Zz,node1,node2):
    i=0                        
    while len(Zz)>2:
        if i!=node1 and i!=node2:
            Zz=MeasureQ(Zz,i)
            if i<node1:
                node1-=1
            if i<node2:
                node2-=1
        else:
            i+=1    
    return Zz

def MeasAllP(Zz,node1,node2):
    i=0                        
    while len(Zz)>2:
        if i!=node1 and i!=node2:
            Zz=MeasureP(Zz,i)
            if i<node1:
                node1-=1
            if i<node2:
                node2-=1
        else:
            i+=1    
    return Zz

def QterminalsPrest(graph, node1,node2):
    Zz=graphState(graph,1,0)
    i=0
    while i<len(Zz):
        if i!=node1 and i!=node2 and graph.degree(i)==1:
            Zz=MeasureQ(Zz,i)
            if i<node1:
                node1-=1
            if i<node2:
                node2-=1
        i+=1
    return MeasAllP(Zz,node1,node2)
            

def findHub(graph):
    return sorted(graph.degree, key=lambda x: x[1], reverse=True)[0][0]

def findFurthestNodes(graph):
    buff=0
    alice=0
    bob=1
    for i in range(len(graph)-1):
        for j in range(i+1,len(graph)):
            lll=nx.shortest_path_length(graph,i,j)
            if lll>buff:
                buff=lll
                alice=i
                bob=j
    return alice,bob

def findFurthestNodeFromNode(graph,node):
    buff=0
    if node==0:
        alice=1
    else:
        alice=0
    for i in range(len(graph)-1):
        lll=nx.shortest_path_length(graph,i,node)
        if lll>buff:
            buff=lll
            alice=i
    return alice                    


     
def trivialRouting(graph,alice,bob,cutoff,*verbose):
    pathList=[]
    i=1
    #while len(pathList)<20 and i<10:
    #    pathList=[]
    #    for path in nx.all_simple_paths(graph, source=alice, target=bob,cutoff=i):
    #        pathList.append(path)
    #    i+=1
    CO=cutoff
    for path in nx.all_simple_paths(graph, source=alice, target=bob, cutoff=CO):
        pathList.append(path)    
        
    pathList=sorted(pathList, key=len)
    
    if verbose:
        print("There are ", len(pathList), " possible paths of maximum length ", CO)
    Z=graphState(graph,1,0)
    
    for i in range(len(pathList)):
        pathList[i].remove(alice)
        pathList[i].remove(bob)
    
    enta=round(negativity(MeasAllQ(Z,alice,bob)),4)
    if verbose:
        print('The initial entanglement between the nodes ',alice, ' and ', bob, ' is ', enta)

    #Plist.append(pathList[0])
    #Qlist=deepcopy(pathList[1:])
    trylist=[]
    if not not pathList:
        trylist=pathList[0]
#    print('The shortest paths bewteen alice and bob are: \n ', Qlist)
    newA=alice
    newB=bob
    Z=graphState(graph,1,0)
    trylist.sort(reverse=True)
    if not not trylist:
        for i in trylist:
            Z=MeasureP(Z,i)
            if i<newA:
                newA-=1
            if i<newB:
                newB-=1
            
        #    i=0                        
        #    while len(Z)>2:
        #        if i!=newA and i!=newB:
        #            Z=MeasureQ(Z,i)
        #            if i<newA:
        #                newA-=1
        #            if i<newB:
        #                newB-=1
        #        else:
        #            i+=1
    Z=MeasAllQ(Z,newA,newB)        
    enta=round(negativity(Z),3)
            
        
    if verbose:    
        print('The entanglement after the protocol is ', enta)
        
#    Z=graphState(graph,1,0)
#    if verbose:
#        print('The entanglement after measuring all nodes in P is ', Fcoh(MeasAllP(Z,alice,bob)))
    
    return enta


def whichPathRoute(graph,alice,bob,cutoff,*verbose):
    pathList=[]
    goodPaths=[]
    i=1
    #while len(pathList)<20 and i<10:
    #    pathList=[]
    #    for path in nx.all_simple_paths(graph, source=alice, target=bob,cutoff=i):
    #        pathList.append(path)
    #    i+=1
    CO=cutoff
    for path in nx.all_simple_paths(graph, source=alice, target=bob, cutoff=CO):
        pathList.append(path)    
        
    pathList=sorted(pathList, key=len)
    #random.shuffle(pathList)
    if verbose:
        print("There are ", len(pathList), " possible paths of maximum length ", CO)
    Z=graphState(graph,1,0)
    
    for i in range(len(pathList)):
        pathList[i].remove(alice)
        pathList[i].remove(bob)
    enta=round(negativity(MeasAllQ(Z,alice,bob)),4)
    if verbose:
        print('The initial entanglement between the nodes ',alice, ' and ', bob, ' is ', enta)
    counter=0
    
    
    pathCounter=0
    Plist=[[]]
    Qlist=[[]]
    #Plist.append(pathList[0])
    #Qlist=deepcopy(pathList[1:])
    Qlist=deepcopy(pathList)
    
    smallist=[]    
    l=0
    while len(Qlist)>0:    
    #    print('The shortest paths bewteen alice and bob are: \n ', Qlist)
        newA=alice
        newB=bob
        Z=graphState(graph,1,0)
        trylist=deepcopy(Plist)
        #Qlist=sorted(Qlist, key=len)     
        trylist.append(deepcopy(Qlist[0]))
        counter=0
        for smallist in trylist:
            counter+=1
            smallist.sort(reverse=True)
            if not not smallist:
                for i in smallist:
                    Z=MeasureP(Z,i)
                    if i<newA:
                        newA-=1
                    if i<newB:
                        newB-=1
                    if counter==len(trylist):
                        for j in range(len(Qlist)):
                            if i in Qlist[j]:
                                Qlist[j].remove(i)
                        for j in range(len(Qlist)):
                            for k in range(len(Qlist[j])):
                                if Qlist[j][k]>i:
                                    Qlist[j][k]-=1
    #    i=0                        
    #    while len(Z)>2:
    #        if i!=newA and i!=newB:
    #            Z=MeasureQ(Z,i)
    #            if i<newA:
    #                newA-=1
    #            if i<newB:
    #                newB-=1
    #        else:
    #            i+=1
        Z=MeasAllQ(Z,newA,newB)        
        if enta<round(negativity(Z),3):
            Plist.append(deepcopy(trylist[-1]))
            enta=round(negativity(Z),3)
            pathCounter+=1
            goodPaths.append(deepcopy(pathList[l]))
            if verbose:
                print("Path ", pathList[l], " improves the entanglement to ", enta)
        Qlist.remove(Qlist[0])
        l+=1
    if verbose:    
        print('The entanglement after the protocol is ', enta)
        
    return goodPaths


def routing(graph,alice,bob,cutoff,*verbose):
    pathList=[]
    i=1
    #while len(pathList)<20 and i<10:
    #    pathList=[]
    #    for path in nx.all_simple_paths(graph, source=alice, target=bob,cutoff=i):
    #        pathList.append(path)
    #    i+=1
    CO=cutoff
    for path in nx.all_simple_paths(graph, source=alice, target=bob, cutoff=CO):
        pathList.append(path)    
        
    pathList=sorted(pathList, key=len)
    #random.shuffle(pathList)
    if verbose:
        print("There are ", len(pathList), " possible paths of maximum length ", CO)
    Z=graphState(graph,1,0)
    
    for i in range(len(pathList)):
        pathList[i].remove(alice)
        pathList[i].remove(bob)
    TOT=len(pathList)
    enta=round(negativity(MeasAllQ(Z,alice,bob)),4)
    if verbose:
        print('The initial entanglement between the nodes ',alice, ' and ', bob, ' is ', enta)
    counter=0
    
    
    pathCounter=0
    Plist=[[]]
    Qlist=[[]]
    #Plist.append(pathList[0])
    #Qlist=deepcopy(pathList[1:])
    Qlist=deepcopy(pathList)
    
    smallist=[]    
    l=0
    while len(Qlist)>0:    
    #    print('The shortest paths bewteen alice and bob are: \n ', Qlist)
        newA=alice
        newB=bob
        Z=graphState(graph,1,0)
        trylist=deepcopy(Plist)
        #Qlist=sorted(Qlist, key=len)     
        trylist.append(deepcopy(Qlist[0]))
        counter=0
        for smallist in trylist:
            counter+=1
            smallist.sort(reverse=True)
            if not not smallist:
                for i in smallist:
                    Z=MeasureP(Z,i)
                    if i<newA:
                        newA-=1
                    if i<newB:
                        newB-=1
                    if counter==len(trylist):
                        for j in range(len(Qlist)):
                            if i in Qlist[j]:
                                Qlist[j].remove(i)
                        for j in range(len(Qlist)):
                            for k in range(len(Qlist[j])):
                                if Qlist[j][k]>i:
                                    Qlist[j][k]-=1

        Z=MeasAllQ(Z,newA,newB)        
        if enta<round(negativity(Z),3):
            Plist.append(deepcopy(trylist[-1]))
            enta=round(negativity(Z),3)
            pathCounter+=1
            if verbose:
                print("Path ", pathList[l], " improves the entanglement to ", enta)
        Qlist.remove(Qlist[0])
        l+=1
    if verbose:    
        print('The entanglement after the protocol is ', enta)
        
    RR=pathCounter/TOT
    return enta,RR

def AllinOne(graph, alice, bob):
    ddd=eee=ttt=ppp=nsp=pr=0
    if alice!=bob:
        ddd=nx.shortest_path_length(graph,alice,bob)
        eee,pr=routing(graph,alice,bob,ddd)
        ttt=trivialRouting(graph,alice,bob,ddd)
        ppp=negativity(QterminalsPrest(graph,alice,bob))
        nsp=len(list(nx.all_simple_edge_paths(graph, source=alice, target=bob, cutoff=ddd)))
    return ddd,eee,ttt,ppp,nsp,pr

#############################################################################
N=20

##complex##

# graph=nx.barabasi_albert_graph(N,4,seed=103)
#graph=nx.random_internet_as_graph(N,seed=109)
#graph=nx.random_geometric_graph(N, 0.125)
#graph=nx.erdos_renyi_graph(N,0.04,seed=108)
#graph=nx.watts_strogatz_graph(N,4,1,seed=12345)
graph=nx.duplication_divergence_graph(N, 0.4,seed=100)

#############################################################################


num_cores = multiprocessing.cpu_count()
x=[i for i in graph.nodes()]#[1:]

inputs=tqdm(x)
Alice=findHub(graph)

start_time = time.time()
results_P=Parallel(n_jobs=num_cores,verbose=5)(delayed(AllinOne)(graph,Alice,i) for i in inputs)

print("Elapsed time in parallel: ", time.time()-start_time )

DIAM,ENT,TENT,PENT,NSP,pathRatio=np.array(list(zip(*results_P)))


#############################################################################


mapping={Alice:0,0:Alice}
# graph=nx.relabel_nodes(graph,mapping)

#DD=[int(i) for _,i in sorted(zip(ENT,DIAM))]


EEE=ENT
TTT=TENT

sorter=DIAM

colors=DIAM
d = dict(graph.degree)


zipped=sorted(zip(sorter,-NSP,pathRatio,x,ENT,TENT,PENT))
nah,noh,RRR,xxx,y1,y2,yp=zip(*zipped)
# xxx = [int(i) for _,i in sorted(zip(sorter,x))]
mapping2=dict(zip(xxx,x))
# graph=nx.relabel_nodes(graph,mapping2)

bestNode=int(np.argmax(ENT))
maxDiff=int(np.argmax(ENT-TENT))

subBestNodeList=[bestNode]
subMaxDiffList=[maxDiff]
subNodeList=[Alice,bestNode,maxDiff]

spn=whichPathRoute(graph,Alice,maxDiff,nx.shortest_path_length(graph,Alice,maxDiff))
for path in spn:
    for i in path:
        if not any(i==item for item in subMaxDiffList):
            subMaxDiffList.append(i)
        check=any(i==item for item in subNodeList)
        if not check:
            subNodeList.append(i)


spn2=whichPathRoute(graph,Alice,bestNode,nx.shortest_path_length(graph,Alice,bestNode))
for path in spn2:
    for i in path:
        if not any(i==item for item in subBestNodeList):
            subBestNodeList.append(i)
        check=any(i==item for item in subNodeList)
        if not check:
            subNodeList.append(i)
        
        
subgraph=graph.subgraph(subNodeList)
#pos = nx.shell_layout(subgraph)
#pos=nx.kamada_kawai_layout(subgraph)
#pos=graphviz_layout(subgraph, prog='twopi',root=alice)

sp2=nx.all_simple_edge_paths(subgraph, source=Alice, target=bestNode, cutoff=nx.shortest_path_length(graph,Alice,bestNode))
loe2=[]
for edge in sp2:
    for i in edge:
        loe2.append(i)

sp=nx.all_simple_edge_paths(subgraph, source=Alice, target=maxDiff, cutoff=nx.shortest_path_length(graph,Alice,maxDiff))
loe=[]
for edge in sp:
    for i in edge:
        loe.append(i)
        
#############################################################################
        
nodeSz=40

plt.figure( dpi=300,figsize=(60,60))
# pos = nx.spring_layout(graph)
pos=graphviz_layout(graph, prog='twopi',root=Alice)

# pos = nx.circular_layout(graph)

ec = nx.draw_networkx_edges(graph, pos, alpha=0.5,width=0.8,edge_color='steelblue')
nc = nx.draw_networkx_nodes(graph, pos, node_color=colors,linewidths=0.5, node_size=[v * nodeSz for v in d.values()],edgecolors='blue', vmin=0,vmax=max(colors)+1, cmap=plt.cm.get_cmap('Blues', max(colors)+1))





nx.draw_networkx_edges(graph, pos,edgelist=loe2, alpha=1,width=3,edge_color='green')
nx.draw_networkx_edges(graph, pos,edgelist=loe, alpha=1,width=2,edge_color='r')


nx.draw_networkx_nodes(subgraph, pos,linewidths=2,node_size=[v*nodeSz for _,v in graph.degree(subBestNodeList)],nodelist=subBestNodeList,edgecolors='green',node_color='none')

nx.draw_networkx_nodes(subgraph, pos,linewidths=1,node_size=[v*nodeSz for _,v in graph.degree(subMaxDiffList)],nodelist=subMaxDiffList,edgecolors='red',node_color='none')





      
        

nx.draw_networkx_nodes(graph, pos,linewidths=5,node_size=graph.degree(bestNode)*nodeSz,nodelist=[bestNode],edgecolors='green',node_color='none')
nx.draw_networkx_nodes(graph, pos,linewidths=3.5,node_size=graph.degree(maxDiff)*nodeSz,nodelist=[maxDiff],edgecolors='red',node_color='none')


nx.draw_networkx_nodes(graph, pos,linewidths=3.5,node_size=graph.degree(Alice)*nodeSz,nodelist=[Alice],edgecolors='red',node_color='none')
nx.draw_networkx_labels(graph, pos,labels=mapping2,font_color='white',font_size=2)
nx.draw_networkx_labels(graph, pos,labels={Alice:0},font_color='red')
# cbar=plt.colorbar(nc,ticks=range(int(max(colors)+1)))
# cbar.set_label(label='Distance from A',size=30)
# cbar.ax.tick_params(labelsize=30)
plt.axis("off")
plt.show()


#############################################################################

plt.figure()

fontsize=18
plt.rcParams.update({'font.size': fontsize})

markerline, stemlines, baseline = plt.stem(x[1:],RRR[1:])
plt.setp(stemlines, color='lightgrey')
plt.setp(stemlines, 'linestyle')
plt.setp(stemlines, 'linewidth', 1)
plt.setp(markerline, markersize = 0)
plt.setp(baseline,visible=False)


markerline, stemlines, baseline = plt.stem(x[1:],y1[1:], markerfmt='o',  label=r'$\mathcal{N}$ Routing')
plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
plt.setp(stemlines, 'linestyle')
plt.setp(stemlines, 'linewidth', 0.5)
plt.setp(markerline, markersize = 0.8)
plt.setp(baseline,visible=False)


markerline, stemlines, baseline = plt.stem(x[1:],y2[1:], markerfmt='o', label=r'$\mathcal{N}$ shortest')
plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
plt.setp(stemlines, 'linestyle')
plt.setp(stemlines, 'linewidth', 0.35)
plt.setp(markerline, markersize = 0.55)
plt.setp(baseline,visible=False)

markerline, stemlines, baseline = plt.stem(x[1:],yp[1:], markerfmt='o', label=r'$\mathcal{N}$ All P')
plt.setp(stemlines, 'color', plt.getp(markerline,'color'))
plt.setp(stemlines, 'linestyle')
plt.setp(stemlines, 'linewidth', 0.15)
plt.setp(markerline, markersize = 0.3)
plt.setp(baseline,visible=False)

plt.plot(x,np.ones(N)*np.average(ENT),label=r'$\langle\mathcal{N}\rangle$ Routing',linestyle='dashed')
plt.plot(x,np.ones(N)*np.average(TENT),label=r'$\langle\mathcal{N}\rangle$ shortest',linestyle='dashed')
plt.plot(x,np.ones(N)*np.average(PENT),label=r'$\langle\mathcal{N}\rangle$ All P',linestyle='dashed')
plt.legend(loc=(0.5,0.4))
plt.ylabel('$\mathcal{N}[\sigma_p]$') 
plt.xlabel('Node') 
plt.grid()
plt.grid(b=True, which='minor', color='gray', linestyle='--',linewidth=0.5)
plt.tight_layout()

plt.xlim([0, N])

ax = plt.gca()
#ax.xaxis.set_minor_locator(MultipleLocator(1))
ax.xaxis.grid(True, which='minor')
plt.xticks(np.arange(0,N+1,100))

plt.scatter(x,y1,c=sorted(DIAM),cmap=plt.cm.get_cmap('Blues', max(colors)+1),zorder=10,s = 0.05)
plt.colorbar(nc,label='Distance from A',ticks=range(int(max(colors)+1)))

plt.show()


#############################################################################


#############################################################################

plt.figure()

nodeSz=55



#pos = nx.spring_layout(subgraph,k=5,iterations=500)
pos=graphviz_layout(subgraph, prog='twopi',root=Alice)

#pos = nx.shell_layout(subgraph)
#pos=nx.kamada_kawai_layout(subgraph)
#pos=graphviz_layout(subgraph, prog='twopi',root=alice)


nx.draw_networkx_edges(subgraph, pos,edgelist=loe2, alpha=1,width=3,edge_color='green')


        
        

nx.draw_networkx_edges(subgraph, pos,edgelist=loe, alpha=1,width=2,edge_color='r')




nx.draw_networkx_edges(subgraph, pos,alpha=0.3)
nx.draw_networkx_nodes(subgraph, pos,linewidths=2,node_size=[v*nodeSz for _,v in graph.degree(subBestNodeList)],nodelist=subBestNodeList,edgecolors='green',node_color='white')

nx.draw_networkx_nodes(subgraph, pos,linewidths=1,node_size=[v*nodeSz for _,v in graph.degree(subMaxDiffList)],nodelist=subMaxDiffList,edgecolors='red',node_color='white')


nx.draw_networkx_nodes(subgraph, pos,linewidths=3.5,node_size=graph.degree(Alice)*nodeSz,nodelist=[Alice],edgecolors='red',node_color='white')
nx.draw_networkx_nodes(graph, pos,linewidths=5,node_size=graph.degree(bestNode)*nodeSz,nodelist=[bestNode],edgecolors='green',node_color='white')
nx.draw_networkx_nodes(graph, pos,linewidths=3.5,node_size=graph.degree(maxDiff)*nodeSz,nodelist=[maxDiff],edgecolors='red',node_color='white')
for node in subNodeList:
    nx.draw_networkx_labels(subgraph, pos,labels={node:mapping2[node]},font_color='black')
plt.axis("off")

plt.show()
