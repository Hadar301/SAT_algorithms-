# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 14:07:51 2021

@author: hcohe
"""


import numpy as np
import pandas as pd
import copy
import time
import sys
import threading


sys.path.insert(0, 'C:\\Users\\hcohe\\Desktop\\codes\\SurveyProp-ver3')
from SurveyProp_classes import *

global_lock_f1 = threading.Lock()


T = 20 #number of rounds
res_dict={}

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

def RandomSAT_test(n,c, test_it, algorithm):
    global T  
    max_iters = 1000
    k = 3 #number of literals per clause 
    m = np.ceil(n*c)
    m = int(m)

    res_arr = np.zeros(11)
    
    for i in range(T//10):
        print("------n={}--c={}---test segment# {}---itertation# {}---\n".format(n,c,test_it,i))
        prop = randomKSAT(n,m,k,max_iters)
        
        start  = time.process_time()
        
        if(algorithm == 'WP'):
            prop.warning_id()
            res_arr += test_results(prop,start)
        elif(algorithm == 'BP'):
            #prop.survey_id_bp()
            prop.belief_prop()
            res_arr += test_results(prop,start)
        elif(algorithm == 'SP'):
            prop.surveyID()
            res_arr += test_results(prop,start)

        del prop
        prop = None

        
    print("------test segment# {} for n={} and c={} for algorithm {} has ended------".format(test_it,n,c, algorithm))
    return res_arr 



def test_results(prop, start):
    SAT_counter = 0
    runtime = 0
    hamming_distnace = 0
    dont_care_conter = 0
    iteration_counter = 0
    num_of_SAT_clauses = 0
    
    absent_literals = 0
    majorityVote_sat_counter = 0
    majorityVote_hamming_distnace = 0
    majorityVote_num_of_SAT_clauses = 0
    
    WP_contradiction_counter = 0
    
    lit_dict, result_val = prop.validateFinalAssignmemt()
    #count number of dont cares
    for key in lit_dict:
        if(lit_dict[key] == "Don't Care" and (prop.majority_vote_dictionary[key] !=0 or prop.majority_vote_dictionary[-key] !=0)):
            dont_care_conter = dont_care_conter +1
    #assign dont cares with random values and check the validity of the literals assignmnet again
    for i in range(len(prop.assignment)):
        if(prop.assignment.astype(int)[i] == 0):
            prop.assignment.astype(int)[i] = np.random.choice([-1,1], 1, p=[0.5, 0.5])
    lit_dict, result_val = prop.validateFinalAssignmemt()
                
    #calculate run time            
    runtime = runtime + (time.process_time() - start)
        
    if(prop.SAT_validation == True):
        SAT_counter += 1
        
    #hamming_distnace += calc_hamming(prop.literal_assignment.astype(int), prop.assignment.astype(int))
    iteration_counter += prop.iteration_counter
    num_of_SAT_clauses += prop.num_of_SAT_clauses
    
    absent_literals += absentLiteralCounter(prop.majority_vote_dictionary) 
    if(prop.SAT_validation_majority == True):
        majorityVote_sat_counter += 1
    majorityVote_num_of_SAT_clauses += prop.num_of_SAT_clauses_majority
    #majorityVote_hamming_distnace += calc_hamming(prop.literal_assignment, prop.majority_vote_result.astype(int))
    
    WP_contradiction_counter = prop.wp_contradiction_counter
    
    return np.array([SAT_counter, num_of_SAT_clauses, hamming_distnace, runtime, dont_care_conter, iteration_counter, absent_literals, 
                     majorityVote_sat_counter, majorityVote_num_of_SAT_clauses, majorityVote_hamming_distnace, WP_contradiction_counter])



def test_flow(n,c, test_it, algorithm):
    res_arr = []
    res_arr = RandomSAT_test(n,c, test_it, algorithm)
    accumulate_results(n, c, res_arr, algorithm)

    
        

def accumulate_results(n, c, res_arr, algorithm):
    global res_dict
    
    '''key_wp = (n, c, 'WP')
    key_bp = (n, c, 'BP')
    key_sp = (n, c, 'SP')'''
    key = (n, c, algorithm)
    
    while(global_lock_f1.locked()):
        continue
    global_lock_f1.acquire()
    
    if(key in res_dict):
        res_dict[key] += res_arr
    elif(key not in res_dict):
        res_dict[key] = res_arr
        
    
    global_lock_f1.release()
    


def parse_results():
    global res_dict
    global T
    
    
        
    df = pd.DataFrame(columns=['N and C', 'SAT Percentage', 'Average Percentage of SAT Clauses', 'Average Percent Hamming distance',
                                  'Average Runtime', 'Average Percent Dont Care', 'Average Number of Iterations', 'Average number of absent literals',
                                  'SAT Percentage (MV)', 'Average Percentage of SAT Clauses(MV)', 'Average Percent Hamming distance (MV)', 
                                  'Average Number of Contradictions (WP)'])
    
    for key in res_dict:
        n = key[0]
        c = key[1]
        algorithm = key[2]
        m = np.ceil(n*c)
        file_name = 'randomSAT_'+ str(algorithm)+'_n_'+str(n)+'.csv'       

        res_arr = res_dict[key]/T
        df = df.append({'N and C' : (n,c), 'SAT Percentage': res_arr[0]*100, 'Average Percentage of SAT Clauses':100*res_arr[1]/m, 
                                  'Average Percent Hamming distance': 100*res_arr[2]/n, 'Average Runtime': res_arr[3]/3600, 
                                  'Average Percent Dont Care': res_arr[4]/n, 'Average Number of Iterations': res_arr[5], 
                                  'Average number of absent literals': res_arr[6]/n, 'SAT Percentage (MV)': res_arr[7]*100, 
                                  'Average Percentage of SAT Clauses(MV)': 100*res_arr[8]/m, 'Average Percent Hamming distance (MV)': 100*res_arr[9]/n, 
                                  'Average Number of Contradictions (WP)': res_arr[10]/n}, ignore_index=True)
            
        df.to_csv(file_name)



def main(n, algorithm):
    np.random.seed(12345)
    global T
    print("------------Running test for n={}, With Algoritm {} -------\n".format(n, algorithm))

    n = int(n)
    c=[5, 10] #,15,20,25]#,30,40]
    thread_list = []
    thread_ratio = T//10
    
    try:
        start = time.process_time()
        for i in range(T//thread_ratio):
            for alpha in c:
                thread = threading.Thread(target= test_flow, args=(n, alpha, i, algorithm))
                thread_list.append(thread)
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()
          
        end = time.process_time()
    except Exception as err:
        print("Error while running a thread")
        print(err)

    parse_results()
    print("Total time of {} minutes".format((end-start)/60))


if __name__ == "__main__":
    n = int(sys.argv[1])
    algorithm = str(sys.argv[2])

    main(n, algorithm)