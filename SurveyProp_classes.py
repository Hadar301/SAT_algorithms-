# -*- coding: utf-8 -*-
"""
Created on Sun Oct 17 12:20:04 2021

@author: Hadar Cohen
"""


import numpy as np
import networkx as nx
from random import shuffle
import random

import time
import math
import pandas as pd



#import matplotlib.pyplot as plt
#from tqdm import tqdm

mainVars = {}

class randomKSAT(object):
    
    def __init__(self, N, M, K, max_iter, eps=1e-3, rand_assignment = False, verbose=False):
        super(randomKSAT, self)
        
        # General features
        self.N = N
        self.M = M
        self.K = K
        self.max_iter = max_iter
        self.eps = eps
        self.rand_assignment = rand_assignment
        self.verbose = verbose
        
        #Array of literals per clause for result validation
        self.literals_per_caluse = []
        #Array that tells us if a literal is True or False in a clause
        self.literals_per_caluse_T_or_F = []
        
        self.majority_vote_dictionary = self.initializeDictionary()
        
        self.warnings_dictionary = self.initializeDictionary()
        
        
        self.num_of_additional_clauses_for_anomaly = 0
        
        self.graph = self.initialize_graph()
        self.dgraph = self.graph.copy()
        
        self.majority_vote_result = self.MajorityVoteSolver()
        
        self.WPstatus = None
        self.SPstatus = None
        self.sat = None
        self.SPstatus = None
        # for warning propagation
        self.H = np.zeros(self.N)
        self.U = np.zeros((self.N, 2))
        # for survey propagation
        self.W = np.zeros((self.N, 3))
        
        self.iteration_counter = 0
        self.num_of_SAT_clauses = self.M
        self.SAT_validation = None
        
        self.num_of_SAT_clauses_majority = self.M
        self.SAT_validation_majority = None
        self.majorityVoteValidation()

        self.wp_contradiction_counter = 0
        
        self.literal_assignment = np.zeros(self.N)
        
        
        
        
        if(self.rand_assignment == False):
            self.assignment = np.zeros(self.N)
        else:
            self.assignment = np.random.choice([-1,1], size=self.N, p=[0.5, 0.5])
                        
        
        if(self.verbose):
            print(self.dgraph.edges())

        

    
    
    
    ###########################################################################
    # Majority Vote
    ###########################################################################        
    
    def MajorityVoteSolver(self):
        res = np.zeros(self.N).astype('int8')
        for key in self.majority_vote_dictionary:
            if(key < 1):
                Negated_literal_value = self.majority_vote_dictionary[key]
                Un_Negated_literal_value = self.majority_vote_dictionary[-key]
                
                if(Negated_literal_value > Un_Negated_literal_value):
                    res[-key] = -1 #False
                else:
                    res[-key] = 1 #True
            else:
                break            
        return res

    
    ###########################################################################
    # WARNING PROPAGATION
    ###########################################################################   
    def warning_prop(self):
        for t in range(self.max_iter):
            d = set(nx.get_edge_attributes(self.dgraph, 'u').items())
            self.wp_update()
            d_ = set(nx.get_edge_attributes(self.dgraph, 'u').items())
            self.iteration_counter += 1
            if d == d_:
                self.WPstatus = 'CONVERGED'
                return
        self.WPstatus = 'UNCONVERGED'

        
    def wp_update(self):
        L = list(self.dgraph.edges())
        shuffle(L)
        for (i,a) in L:
            # Compute cavity fields
            for j in self.dgraph.neighbors(a):
                if i == j:
                    continue
                self.dgraph[j][a]['h'] = 0
                for b in self.dgraph.neighbors(j):
                    if b == a:
                        continue
                    self.dgraph[j][a]['h'] += self.dgraph[j][b]['u'] * self.dgraph[j][b]['J']

            # Compute warnings
                self.dgraph[i][a]['u'] = 1
                for j in self.dgraph.neighbors(a):
                    if i == j:
                        continue
                    self.dgraph[i][a]['u'] *= np.heaviside(- self.dgraph[j][a]['h'] * self.dgraph[j][a]['J'], 0)


                        
    def warning_id(self):
        self.assignment = np.random.choice([-1,1], size=self.N, p=[0.5, 0.5])
        while(self.WPstatus == None): #np.any(self.assignment == 0):
            self.warning_prop()
            if self.WPstatus == 'UNCONVERGED':
                return
            self.wid_localfield()
            if self.sat == 'UNSAT':
                return
            self.wid_assignment()
            self.decimate_graph()
            if(self.verbose):
                print(self.assignment)
                print(self.H)
                print("NODES = ", self.dgraph.number_of_nodes())
                print("EDGES = ", self.dgraph.number_of_edges())
                print(self.dgraph.edges())
        self.dgraph = self.graph.copy()
    
    def wid_localfield(self):
        # Compute local fields and contradiction numbers
        for i in range(self.N):
            if i not in self.dgraph.nodes():
                continue
            self.H[i] = 0
            self.U[i] = 0
            for a in self.dgraph.neighbors(i):
                self.H[i] -=  self.dgraph[i][a]['u'] * self.dgraph[i][a]['J']
                self.U[i, int(np.heaviside(self.dgraph[i][a]['J'], 0))] += self.dgraph[i][a]['u']
        c = self.U[:,0] * self.U[:,1]
        self.wp_contradiction_counter = sum(c)
        if np.amax(c) > 0:
            self.sat = 'UNSAT'
            return
        self.sat = 'SAT'
    
    def wid_assignment(self):
        # Determine satisfiable assignment
        mask = np.array([i in self.dgraph.nodes() for i in range(self.N)])
        if np.any(self.H[mask] != 0):
            self.assignment[(self.H > 0) & mask] = 1
            self.assignment[(self.H < 0) & mask] = -1
        else:
            p = np.argmax(self.H == 0)
            self.H[p] = 1
            self.assignment[p] = 1
    
    
    ###########################################################################
    # Belief PROPAGATION
    ###########################################################################
 
    def multiplicationForBP(self, values_arr):
        result = 1
        for value in values_arr:
            result *= (1-value)
        return result
    
    def belief_prop(self):
        # Initialize deltas on the edges
        for (i,a) in self.dgraph.edges():
            self.dgraph[i][a]['delta'] = np.random.rand(1)
        
        #Iteration
        for t in range(self.max_iter):    
            d = np.array(list(nx.get_edge_attributes(self.dgraph, 'delta').values()))
            self.bp_update()
            d_ = np.array(list(nx.get_edge_attributes(self.dgraph, 'delta').values()))
            self.iteration_counter +=1
            if np.all(np.abs(d - d_) < self.eps):
                self.bp_assignment()
                return
        self.SPstatus = 'UNCONVERGED'
        
    def bp_update(self):
        L = list(self.dgraph.edges())
        shuffle(L)
        for (i,a) in L:
            # Compute cavity fields
            for j in self.dgraph.neighbors(a):
                if i == j:
                    continue
                prod_tmp = np.ones(2)
                for b in self.dgraph.neighbors(j):
                    if b == a:
                        continue
                    p = int(np.heaviside(self.dgraph[j][a]['J'] * self.dgraph[j][b]['J'], 0))
                    prod_tmp[p] *= (1 - self.dgraph[j][b]['delta'])
                self.dgraph[j][a]['P_u'] = prod_tmp[1]
                self.dgraph[j][a]['P_s'] = prod_tmp[0]
                        
            # Compute warnings
            self.dgraph[i][a]['delta'] = 1
            for j in self.dgraph.neighbors(a):
                if i == j:
                    continue
                tot = (self.dgraph[j][a]['P_u'] + self.dgraph[j][a]['P_s'])
                if tot == 0:
                    self.dgraph[i][a]['delta'] = 1
                    break
                p = self.dgraph[j][a]['P_u'] / tot
                self.dgraph[i][a]['delta'] *= p
                
    def bp_assignment(self):
        pos_dict = {}
        neg_dict = {}            
                
        for (node,clause) in nx.get_edge_attributes(self.dgraph, 'delta'):
            sign = self.dgraph[node][clause]['J']
            if(sign == -1):
                #append to pos_dict
                if(node in pos_dict):
                    pos_dict[node].append(self.dgraph[node][clause]['delta'])
                else:
                    pos_dict[node] = [self.dgraph[node][clause]['delta']]
            elif(sign == 1):
                #append to neg_dict
                if(node in neg_dict):
                    neg_dict[node].append(self.dgraph[node][clause]['delta'])
                else:
                    neg_dict[node] = [self.dgraph[node][clause]['delta']]

        for key in range(self.N):
            numerator = 1
            denominator = 1
            if(key in pos_dict):
                delta_arr1 = pos_dict[key]
                numerator = self.multiplicationForBP(delta_arr1)
                denominator = numerator
            else:
                numerator = 0
                        
            if(key in neg_dict):
                delta_arr2 = neg_dict[key]
                denominator += self.multiplicationForBP(delta_arr2)
                    
            if(denominator == 0):
                res = 1
            else:
                res = numerator/denominator

            #self.assignment[key] = np.random.choice([-1,1], size=1, p=[res,1-res])#p=[1-res, res])
            
            '''
            if the probability to be FLASE is greater than 0.6, assign the literal to -1 (flase)
            if the probability to be TRUE is greater than 0.6, assign the literal to 1 (true)
            else, assign 0 (don't care) 
            '''
            if(res > 0.6):
                self.assignment[key] = -1 
            elif (1-res > 0.6):
                self.assignment[key] = 1
            else:
                self.assignment[key] = 0
             
        

    ###########################################################################
    # SURVEY PROPAGATION
    ###########################################################################
 



    def sp_update(self):
        L = list(self.dgraph.edges())
        shuffle(L)
        for (i,a) in L:
            # Compute cavity fields
            for j in self.dgraph.neighbors(a):
                if i == j:
                    continue
                prod_tmp = np.ones(2)
                for b in self.dgraph.neighbors(j):
                    if b == a:
                        continue
                    p = int(np.heaviside(self.dgraph[j][a]['J'] * self.dgraph[j][b]['J'], 0))
                    prod_tmp[p] *= (1 - self.dgraph[j][b]['delta'])
                self.dgraph[j][a]['P_u'] = (1 - prod_tmp[0]) * prod_tmp[1]
                self.dgraph[j][a]['P_s'] = (1 - prod_tmp[1]) * prod_tmp[0]
                self.dgraph[j][a]['P_0'] = prod_tmp[0] * prod_tmp[1]
                self.dgraph[j][a]['P_c'] = (1-prod_tmp[0]) * (1-prod_tmp[1])
                        
            # Compute warnings
            self.dgraph[i][a]['delta'] = 1
            for j in self.dgraph.neighbors(a):
                if i == j:
                    continue
                tot = (self.dgraph[j][a]['P_u'] + self.dgraph[j][a]['P_s'] + self.dgraph[j][a]['P_0'])# + self.dgraph[j][a]['P_c'])
                if tot == 0:
                    self.dgraph[i][a]['delta'] = 1
                    break
                p = self.dgraph[j][a]['P_u'] / tot
                self.dgraph[i][a]['delta'] *= p

        
    def sid_localfield(self):
        prod_tmp = np.ones((self.N, 2))
        for i in range(self.N):
            if i not in self.dgraph.nodes():
                continue
            for a in self.dgraph.neighbors(i):
                p = int(np.heaviside(self.dgraph[i][a]['J'], 0))
                prod_tmp[i,p] *= (1 - self.dgraph[i][a]['delta'])
        pi = np.ones((self.N, 4))
        pi[:,0] = (1 - prod_tmp[:,0]) * prod_tmp[:,1] # V plus
        pi[:,1] = (1 - prod_tmp[:,1]) * prod_tmp[:,0]
        pi[:,2] = prod_tmp[:,0] * prod_tmp[:,1]
        #pi[:,3] = (1 - prod_tmp[:,0]) * (1 - prod_tmp[:,1])
        tot = (pi[:,0] + pi[:,1] + pi[:,2])# + pi[:,3])
        pi[(tot == 0),0] = 0
        pi[(tot == 0),1] = 0
        pi[(tot == 0),2] = 0
        tot[tot == 0] = 1
        self.W[:,0] = pi[:,0] / tot
        self.W[:,1] = pi[:,1] / tot
        self.W[:,2] = pi[:,2] / tot
        
        
    def survey_prop(self):
        for (i,a) in self.dgraph.edges():
            self.dgraph[i][a]['delta'] = np.random.rand(1)
        for t in range(self.max_iter):
            d = np.array(list(nx.get_edge_attributes(self.dgraph, 'delta').values()))
            self.sp_update()
            d_ = np.array(list(nx.get_edge_attributes(self.dgraph, 'delta').values()))
            self.iteration_counter +=1
            if(np.all(np.abs(d-d_))<self.eps):
                self.SPstatus = 'CONVERGED'
                return
        self.SPstatus = 'UNCONVERGED'
        
    def surveyID(self):
        self.survey_prop()
        if(self.SPstatus == 'CONVERGED'):
            if(self.nonTrivialSurvey() == True):
                while(self.dgraph.number_of_edges() > 0):
                    self.sid_localfield()
                    p = np.argmax(np.abs(self.W[:,0] - self.W[:,1]))
                    self.assignment[p] = np.sign(self.W[p,0] - self.W[p,1])
                    if(self.W[p,0] == self.W[p,1]):
                        p = int(list(self.dgraph.nodes())[0])
                        self.assignment[p] = np.random.choice([-1,1], size=1, p=[0.5, 0.5])
                    self.decimate_graph()
                return
            else:
                ##need to implement wlaksat (random walk)
                self.assignment = np.random.choice([-1,1], size=self.N, p=[0.5, 0.5])
        else:
            self.SAT_validation = False
            return
    
    def nonTrivialSurvey(self):
        for (i,a) in self.dgraph.edges():
            if(self.dgraph[i][a]['delta'] != 0):
                return True
        return False
        
    ###########################################################################
    # SERVICE FUNCTIONS
    ###########################################################################

    def initialize_graph(self):
        G = nx.Graph()
        G.add_nodes_from(np.arange(self.N + self.M))
        for t in range(self.M):
            B = np.unique(np.random.choice(self.N, self.K))
            self.literals_per_caluse.append(B.tolist())
            J = np.random.binomial(1, .5, size = np.shape(B))
            J[J==0] = -1
            self.literals_per_caluse_T_or_F.append(J.tolist())
            G.add_edges_from([(x, self.N + t) for x in B])
            self.countLiteralInClause(B, J)
            for i in range(len(B)):
                G[B[i]][self.N + t]['J'] = J[i]
                G[B[i]][self.N + t]['u'] = np.random.binomial(1, .5)
                G[B[i]][self.N + t]['h'] = 0
                G[B[i]][self.N + t]['delta'] = np.random.rand(1)
        return G
    
    def initializeDictionary(self):
        return dict.fromkeys(list(range(-self.N+1, self.N)),0)
    
    def countLiteralInClause(self, clause, signs):
        for i in range(len(clause)):
            literal = clause[i]
            if(signs[i] == -1):
                key = literal
            elif(signs[i] == 1):
                key = -1*literal
            self.majority_vote_dictionary[key] +=1
        
            
    
    def decimate_graph(self):
        for i in range(self.N):
            if self.assignment[i] == 0:
                continue
            if i not in self.dgraph.nodes():
                continue
            l = []
            for a in self.dgraph.neighbors(i):
                if self.dgraph[i][a]['J'] * self.assignment[i] == -1:
                    l.append(a)
            for a in l:
                self.dgraph.remove_node(a)
            self.dgraph.remove_node(i)
            
    def check_truth(self):
        l = []
        for a in range(self.N, self.N + self.M):
            for i in self.graph.neighbors(a):
                if self.graph[i][a]['J'] * self.assignment[i] == -1:
                    l.append(a)
                    break
        return len(l) == self.M
    

    def validateFinalAssignmemt(self):
         assignments = self.assignment.astype(int)
         results_arr = []
         literal_dict = {}
         self.num_of_SAT_clauses = self.M + self.num_of_additional_clauses_for_anomaly
         result = False
         for i in range(0, self.N):
            literal_dict[i] = "Don't Care"
         for i in range(0,len(self.literals_per_caluse)):
             
             clause = self.literals_per_caluse[i]
             in_caluse_assignments = self.literals_per_caluse_T_or_F[i]
             
             result_arr = []
             for j in range(0,len(clause)):
                literal_operation = in_caluse_assignments[j] 
                literal = clause[j]
                if(assignments[literal] == 0): #Dont care
                    result_arr.append(True)
                elif(literal_operation == -1 and assignments[literal] == 1):
                    result_arr.append(True)
                elif(literal_operation == -1 and assignments[literal] == -1):
                    result_arr.append(False)
                elif(literal_operation == 1 and assignments[literal] == 1):
                    result_arr.append(False)
                elif(literal_operation == 1 and assignments[literal] == -1):
                    result_arr.append(True)
             results_arr.append(result_arr)       
         for caluse_assignment in results_arr:
             result = False
             for literal_assignment in caluse_assignment:
                 result = result or literal_assignment
             if(result == False):
                self.num_of_SAT_clauses -= 1 
        
        
         for i in range(0, len(assignments)):
            if(assignments[i]==1):
                literal_dict[i] = 'True'
            elif(assignments[i]==-1):
                literal_dict[i] = 'False'
             #print("\nThis expression is SAT the literals values are:\n\n{}".format(literal_dict))
             
         if(self.num_of_SAT_clauses == self.M + self.num_of_additional_clauses_for_anomaly):
             self.SAT_validation = True
         else:
             self.SAT_validation = False
         return literal_dict, result
     
        
    def majorityVoteValidation(self):
        assignments = self.majority_vote_result.astype(int)
        results_arr = []

        for i in range(0,len(self.literals_per_caluse)):
            clause = self.literals_per_caluse[i]
            in_caluse_assignments = self.literals_per_caluse_T_or_F[i]
            result_arr = []
            for j in range(0,len(clause)):
                literal_operation = in_caluse_assignments[j] 
                literal = clause[j]
                if(assignments[literal] == 0): #Dont care
                    result_arr.append(True)
                elif(literal_operation == -1 and assignments[literal] == 1):
                    result_arr.append(True)
                elif(literal_operation == -1 and assignments[literal] == -1):
                    result_arr.append(False)
                elif(literal_operation == 1 and assignments[literal] == 1):
                    result_arr.append(False)
                elif(literal_operation == 1 and assignments[literal] == -1):
                    result_arr.append(True)
            
            results_arr.append(result_arr)       
        for caluse_assignment in results_arr:
             result = False
             for literal_assignment in caluse_assignment:
                 result = result or literal_assignment
             if(result == False):
                self.num_of_SAT_clauses_majority -= 1
        
        if(self.num_of_SAT_clauses_majority == self.M):
            self.SAT_validation_majority = True
        else:
            self.SAT_validation_majority = False
     
    def dontCarePrecentage(self):
         counter = 0 
         assignments = self.assignment.astype(int)
         for literal_assignment in assignments:
             if(literal_assignment == 0):
                 counter = counter + 1
         
         return counter / self.N
 
        
     
class CNF_KSAT(randomKSAT):
        
    def __init__(self, max_iter, filePath, eps=1e-3, rand_assignment = False, verbose=False):
        self.max_iter = max_iter
        self.filePath = filePath
        self.eps = eps
        self.verbose = verbose
        
        #Array of literals per clause for result validation
        self.literals_per_caluse = []
        #Array that tells us if a literal is True or False in a clause
        self.literals_per_caluse_T_or_F = []
        
        self.num_of_additional_clauses_for_anomaly = 0
        
        self.graph = self.initialize_graph()
        
        self.dgraph = self.graph.copy()
        self.WPstatus = None
        self.SPstatus = None
        self.sat = None
        self.rand_assignment = rand_assignment
        # for warning propagation
        self.H = np.zeros(self.N)
        self.U = np.zeros((self.N, 2))
        # for survey propagation
        self.W = np.zeros((self.N, 3))
        
        
        
        if(self.rand_assignment == False):
            self.assignment = np.zeros(self.N)
        else:
            self.assignment = np.random.choice([0,1], size=self.N, p=[0.5, 0.5])
        
        if(self.verbose):
            print(self.dgraph.edges())
    
    def initialize_graph(self):
        G = nx.Graph()
        file = open(self.filePath, 'r')
        Lines = file.readlines()
        count = 0
        for line in Lines:
            currLine = line.strip()
            if currLine[0] == 'c':
                continue
            elif currLine[0] == 'p':
                lineArr = line.split()
                self.N = int(lineArr[2])
                self.M = int(lineArr[3])
                G.add_nodes_from(np.arange(self.N + self.M))
            else:
                lineArr = line.split()
                arrInt = list(map(int,lineArr))
                if 0 in arrInt:
                    arrInt.remove(0)
                B1 = list(map(abs,arrInt))
                B = [item-1 for item in B1]
                J=[]
                for literal in arrInt:
                    if literal > 0:
                        J.append(-1)
                    else:
                        J.append(1)
                self.literals_per_caluse.append(B)
                self.literals_per_caluse_T_or_F.append(J)
                G.add_edges_from([(x, self.N + count) for x in B])
                for i in range(len(B)):
                        G[B[i]][self.N + count]['J'] = J[i]
                        G[B[i]][self.N + count]['u'] = np.random.binomial(1, .5)
                        G[B[i]][self.N + count]['h'] = 0
                        G[B[i]][self.N + count]['delta'] = np.random.rand(1)
                count = count + 1
                
        return G
        

class RandomPlantedSAT(randomKSAT):

    def __init__(self, N, M, K, max_iter, eps=1e-3, verbose=False):
        self.N = N
        self.M = M
        self.K = K
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        
        #Array of literals per clause for result validation
        self.literals_per_caluse = []
        #Array that tells us if a literal is True or False in a clause
        self.literals_per_caluse_T_or_F = []
        
        self.majority_vote_dictionary = self.initializeDictionary()
        
        self.warnings_dictionary = self.initializeDictionary()
        
        #self.initial_literal_assingment = np.random.choice([0, 1], size=self.N, p=[.5, .5])
        self.literal_assignment = np.random.choice([-1,1], size=self.N, p=[0.5, 0.5])
        #print(self.literal_assignment)
        self.num_of_additional_clauses_for_anomaly = 0
        self.graph = self.initialize_graph()
        
        
        self.majority_vote_result = np.array(self.MajorityVoteSolver())
                
        self.dgraph = self.graph.copy()
        self.WPstatus = None
        self.SPstatus = None
        self.sat = None
        self.status = None
        self.assignment = np.zeros(self.N)
        # for warning propagation
        self.H = np.zeros(self.N)
        self.U = np.zeros((self.N, 2))
        # for survey propagation
        self.W = np.zeros((self.N, 3))
        
        self.SAT_validation = None
        self.iteration_counter = 0
        self.num_of_SAT_clauses = self.M
        self.num_of_SAT_clauses_majority = self.M
        self.SAT_validation_majority = None
        
        self.majorityVoteValidation()
        #print(self.num_of_SAT_clauses_majority)
        
        self.wp_contradiction_counter = 0
        
        
        self.literal_warning_flags = np.zeros(self.N)
        
        if(self.verbose):
            print(self.dgraph.edges()) 
    
    def initialize_graph(self):
        G = nx.Graph()
        G.add_nodes_from(np.arange(self.N + self.M))   
        t=0
        while(t<self.M):
            B = np.unique(np.random.choice(self.N, self.K))
            J = np.random.binomial(1, .5, size = np.shape(B))
            J[J==0] = -1
            if(self.approvedClause(B, J) == True):
                #t=t+1
                self.literals_per_caluse.append(B.tolist())
                self.literals_per_caluse_T_or_F.append(J.tolist())
                G.add_edges_from([(x, self.N + t) for x in B])
                self.countLiteralInClause(B, J)

                for i in range(len(B)):
                    G[B[i]][self.N + t]['J'] = J[i]
                    G[B[i]][self.N + t]['u'] = np.random.binomial(1, .5)
                    G[B[i]][self.N + t]['h'] = 0
                    G[B[i]][self.N + t]['delta'] = np.random.rand(1)
                t+=1
            #else:
                #print("unSAT clause, fine new one")
        
        return G
        
    def approvedClause(self, B, J):
        clause_values=[]
        clause_result = False
        
        for i in range(0,len(B)):
            literal = B[i]
            if(self.literal_assignment[literal] == 1 and J[i] == -1):
                clause_values.append(True)
            elif(self.literal_assignment[literal] == 1 and J[i] == 1):
                clause_values.append(False)
            elif(self.literal_assignment[literal] == -1 and J[i] == -1):
                clause_values.append(False)
            elif(self.literal_assignment[literal] == -1 and J[i] == 1):
                clause_values.append(True)
        
        for value in clause_values:
            clause_result = clause_result or value
        return clause_result
    

class RandomPlantedSAT_coreAnomaly(randomKSAT):
    def __init__(self, N, M, K, max_iter, eps=1e-3, verbose=False):
        self.N = N
        self.M = M
        self.K = K
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose
        
        #Array of literals per clause for result validation
        self.literals_per_caluse = []
        #Array that tells us if a literal is True or False in a clause
        self.literals_per_caluse_T_or_F = []
        
        self.majority_vote_dictionary = self.initializeDictionary()
        
        self.warnings_dictionary = self.initializeDictionary()
        
        #self.initial_literal_assingment = np.random.choice([0, 1], size=self.N, p=[.5, .5])
        self.literal_assignment = np.random.choice([-1,1], size=self.N, p=[0.5, 0.5])
        #print(self.literal_assignment)
        self.num_of_additional_clauses_for_anomaly = 150
        self.graph = self.initialize_graph()
        
        
        self.majority_vote_result = np.array(self.MajorityVoteSolver())
                
        self.dgraph = self.graph.copy()
        self.WPstatus = None
        self.SPstatus = None
        self.sat = None
        self.status = None
        self.assignment = np.zeros(self.N)
        # for warning propagation
        self.H = np.zeros(self.N)
        self.U = np.zeros((self.N, 2))
        # for survey propagation
        self.W = np.zeros((self.N, 3))
        
        self.SAT_validation = None
        self.iteration_counter = 0
        self.num_of_SAT_clauses = self.M
        self.num_of_SAT_clauses_majority = self.M
        self.SAT_validation_majority = None
        
        self.majorityVoteValidation()
        #print(self.num_of_SAT_clauses_majority)
        
        self.wp_contradiction_counter = 0
        
        
        
        self.literal_warning_flags = np.zeros(self.N)
        
        if(self.verbose):
            print(self.dgraph.edges()) 
    
    def initialize_graph(self):
        G = nx.Graph()
        if(self.N >=50):
            G.add_nodes_from(np.arange(self.N + self.M + self.num_of_additional_clauses_for_anomaly))
        t=0
        anomaly_nodes_counter = self.M 
        while(t<self.M):
            B = np.unique(np.random.choice(self.N, self.K))
            J = np.random.binomial(1, .5, size = np.shape(B))
            J[J==0] = -1
            if(self.approvedClause(B, J) == True):
                #t=t+1
                self.literals_per_caluse.append(B.tolist())
                self.literals_per_caluse_T_or_F.append(J.tolist())
                G.add_edges_from([(x, self.N + t) for x in B])
                self.countLiteralInClause(B, J)

                for i in range(len(B)):
                    G[B[i]][self.N + t]['J'] = J[i]
                    G[B[i]][self.N + t]['u'] = np.random.binomial(1, .5)
                    G[B[i]][self.N + t]['h'] = 0
                    G[B[i]][self.N + t]['delta'] = np.random.rand(1)
                t+=1



        if(self.N >= 50):
            support_literals_list = random.sample(list(range(self.N)), 50)
            while(anomaly_nodes_counter < self.M + self.num_of_additional_clauses_for_anomaly):
                counter = 0
                support_litral = random.sample(support_literals_list, 1)[0]
                while(counter<3):
                    B=[]
                    J=[]
                    B.append(support_litral)
                    J = np.random.binomial(1, .5, size = np.shape(B))
                    J[J==0] = -1
            

                    while(self.approvedClause(B, J) != True):
                        if(J[0] == -1):
                            J[0] = 1
                        else:
                            J[0] = -1
                        unsupported_literals = random.sample(support_literals_list, 2)
                        while(support_litral in unsupported_literals):
                            unsupported_literals = random.sample(support_literals_list, 2)
                        temp_J = np.random.binomial(1, .5, size = np.shape(unsupported_literals))
                        temp_J[temp_J==0] = -1
                        while(self.approvedClause(unsupported_literals, temp_J) != False):
                            temp_J = np.random.binomial(1, .5, size = np.shape(unsupported_literals))
                            temp_J[temp_J==0] = -1
                            
                        B.extend(unsupported_literals)
                        J = np.append(J, temp_J)
                        self.literals_per_caluse.append(B)
                        self.literals_per_caluse_T_or_F.append(J.tolist())
                        G.add_edges_from([(x, self.N + anomaly_nodes_counter) for x in B])
                        self.countLiteralInClause(B, J)
                        
                        if(self.approvedClause(B, J) != True):
                            print("!!!!!!!!!!!!!!!!!!!")
                        for i in range(len(B)):
                            G[B[i]][self.N + anomaly_nodes_counter]['J'] = J[i]
                            G[B[i]][self.N + anomaly_nodes_counter]['u'] = np.random.binomial(1, .5)
                            G[B[i]][self.N + anomaly_nodes_counter]['h'] = 0
                            G[B[i]][self.N + anomaly_nodes_counter]['delta'] = np.random.rand(1)
                        anomaly_nodes_counter +=1
                        counter+=1

                        #print(anomaly_nodes_counter)
        #print(len(self.literals_per_caluse))
        return G
        
    def approvedClause(self, B, J):
        clause_values=[]
        clause_result = False
        
        for i in range(0,len(B)):
            literal = B[i]
            if(self.literal_assignment[literal] == 1 and J[i] == -1):
                clause_values.append(True)
            elif(self.literal_assignment[literal] == 1 and J[i] == 1):
                clause_values.append(False)
            elif(self.literal_assignment[literal] == -1 and J[i] == -1):
                clause_values.append(False)
            elif(self.literal_assignment[literal] == -1 and J[i] == 1):
                clause_values.append(True)
        
        for value in clause_values:
            clause_result = clause_result or value
        return clause_result
    
def calc_hamming(vec1, vec2):
    ham_dist = 0
    
    for i in range(len(vec1)):
        if(vec1[i] != vec2[i]):
            ham_dist = ham_dist + 1
    return ham_dist

def absentLiteralCounter(lit_dict):
    counter = 0
    for key in lit_dict:
        if(lit_dict[key] == 0):
            counter +=1
    return counter


def main():
    #np.random.seed(12345)
    #prop=randomKSAT(5,2,3,100)
    '''
    counter = 0
    for i in range(1000):
        prop = RandomPlantedSAT(500,500*5,3,700)
        num_of_absent_literals = absentLiteralCounter(prop.majority_vote_dictionary)
        if(num_of_absent_literals > 0):
            counter+=1
        del prop
    print(counter)'''
    

    prop2 = RandomPlantedSAT(50,50*10,3,100) #randomKSAT(2,20,3,100) #RandomPlantedSAT(3,300,3,1000) 
    #prop2 = copy.deepcopy(prop1)
    #prop3 = copy.deepcopy(prop1)
    
    start = time.process_time()
    #prop2.belief_prop()#warning_id()#survey_id_sp()
    #prop2.surveyID()
    #prop2.belief_prop()
    prop2.warning_id()
    print(prop2.iteration_counter)
    lit_dict, result_val = prop2.validateFinalAssignmemt()
    print(prop2.SAT_validation)
    
    #print("BP Status = ", prop2.BPstatus)
    #print("Satisfiability = ", prop2.sat)
    #print(prop2.check_truth())
    #print(prop2.validateFinalAssignmemt()[0])
    lit_dict, result_val = prop2.validateFinalAssignmemt()
    end = time.process_time()
    #print(prop2.SAT_validation)
    print("Total time of warning propagation {} seconds".format((end-start)))
    #print(prop2.assignment.astype(int))
    #print(prop2.literal_assignment)
    #print(calc_hamming(prop2.literal_assignment, prop2.assignment.astype(int)))
    #print(prop2.assignment.astype(int))

    print("#######################")
    prop3 = RandomPlantedSAT_coreAnomaly(50,50*10,3,100)
    prop3.belief_prop()
    print(prop3.iteration_counter)
    lit_dict, result_val = prop3.validateFinalAssignmemt()
    print(prop3.SAT_validation)
    
    '''
    print(prop1.majority_vote_dictionary)
    print("\n\n")
    start = time.process_time()
    prop1.warning_id()
    print("WP Status = ", prop1.WPstatus)
    print("Satisfiability = ", prop1.sat)
    print(prop1.validateFinalAssignmemt())
    print(prop1.wp_contradiction_counter)
    #print("assignments befor\n {}".format(prop1.assignment.astype(int)))
    for i in range(len(prop1.assignment)):
        if(prop1.assignment.astype(int)[i] == 0):
            prop1.assignment.astype(int)[i] = np.random.choice([-1,1], 1, p=[0.5, 0.5])
    #print("assignments after\n {}".format(prop1.assignment.astype(int)))
    lit_dict, result_val = prop1.validateFinalAssignmemt()
    #print(prop1.validateFinalAssignmemt())
    end = time.process_time()
    print("is this problem SAT? {}".format(prop1.SAT_validation))
    
    print(prop1.assignment)
    print("Total time of warning propagation {} seconds\n".format((end-start)))
    
    #print(calc_hamming(prop1.literal_assignment, prop1.assignment.astype(int)))
    
    

    start = time.process_time()
    prop3.survey_id_sp()
    print("SP Status = ", prop3.SPstatus)
    print("Satisfiability = ", prop3.sat)
    print(prop3.check_truth())
    print(prop3.validateFinalAssignmemt())
    for i in range(len(prop3.assignment)):
        if(prop3.assignment.astype(int)[i] == 0):
            prop3.assignment.astype(int)[i] = np.random.choice([-1,1], 1, p=[0.5, 0.5])
    lit_dict, result_val = prop3.validateFinalAssignmemt()
    print(result_val)
    end = time.process_time()
    print(prop3.SAT_validation)
    print("Total time of survey propagation {} seconds".format((end-start)))
    print(calc_hamming(lit_dict, prop2.assignment.astype(int)))
    
    start = time.process_time()
    prop2.survey_id_bp()
    print("SP Status = ", prop2.SPstatus)
    print("Satisfiability = ", prop2.sat)
    print(prop2.check_truth())
    print(prop2.validateFinalAssignmemt())
    end = time.process_time()
    print(prop2.SAT_validation)
    print("Total time of belief propagation {} seconds".format((end-start)))
    print(calc_hamming(lit_dict, prop3.assignment.astype(int)))
    
    
    

    prop1.majorityVoteValidation()
    print("majority vote made {}% of SAT clauses, and the problem was SAT? {}".format(prop1.num_of_SAT_clauses_majority, prop1.SAT_validation_majority))
    #print(prop1.num_of_SAT_clauses_majority)
    #print(prop1.SAT_validation_majority)

    print(calc_hamming(prop1.literal_assignment, prop1.majority_vote_result.astype(int)))'''

                  
    
if __name__ == "__main__":
    main()


'''
    def survey_id_sp(self):
        max_it = 0
        dont_care_ratio = self.dontCarePrecentage()
        while np.any(self.assignment == 0) and (self.dgraph.number_of_edges() > 0):# and (max_it< self.max_iter):
            #self.iteration_counter +=1
            self.survey_prop()
            if self.SPstatus == 'UNCONVERGED':
                print('UNCONVERGED SID')
                return
            if np.amax(np.array(list(nx.get_edge_attributes(self.dgraph, 'delta').values()))) > self.eps:
#                mask = np.array([i in self.dgraph.nodes() for i in range(self.N)])
                self.sid_localfield()
                p = np.argmax(np.abs(self.W[:,0] - self.W[:,1]))
                self.assignment[p] = np.sign(self.W[p,0] - self.W[p,1])
            else:
                p = list(self.dgraph.nodes())[0]
                p = p.astype(int)
                self.assignment[p] = 1
            self.decimate_graph()
            dont_care_ratio = self.dontCarePrecentage()
            if(self.verbose):
                print(self.assignment)
                print("NODES = ", self.dgraph.number_of_nodes())
                print("EDGES = ", self.dgraph.number_of_edges())
                print(self.dgraph.edges())
                print('\n')
            max_it += 1
        self.dgraph = self.graph.copy()
        if max_it == self.max_iter:
            self.SPstatus = 'UNCONVERGED'
            print("UNCONVERGED SID")
            
            
            
    def survey_prop(self):
        for t in range(self.max_iter):
            d = np.array(list(nx.get_edge_attributes(self.dgraph, 'delta').values()))
            self.sp_update()
            d_ = np.array(list(nx.get_edge_attributes(self.dgraph, 'delta').values()))
            self.iteration_counter +=1
            if(np.all(np.abs(d-d_))<self.eps):
                self.SPstatus = 'CONVERGED'
                return
            self.SPstatus = 'UNCONVERGED'
            
'''