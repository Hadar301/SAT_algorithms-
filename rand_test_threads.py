# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 14:07:51 2021

@author: hcohe
"""

import sys
import time
import threading
import numpy as np

sys.path.insert(0, 'C:\\Users\\hcohe\\Desktop\\codes\\SurveyProp-ver3')

from SurveyProp_classes import *
from main_test_flow import *







def main(n, algorithm):
    np.random.seed(12345)
    global T
    print("------------Running test for n={}, With Algoritm {} -------\n".format(n, algorithm))

    n = int(n)
    c=[1,2] #,15,20,25]#,30,40]
    thread_list = []
    thread_ratio = T//10
    
    try:
        start = time.process_time()
        for i in range(T//thread_ratio):
            for alpha in c:
                thread = threading.Thread(target= test_flow, args=(n, alpha, i, algorithm, randomKSAT))
                thread_list.append(thread)
        for thread in thread_list:
            thread.start()
        for thread in thread_list:
            thread.join()
          
        end = time.process_time()
    except Exception as err:
        print("Error while running a thread")
        print(err)

    parse_results(randomKSAT)
    print("Total time of {} minutes".format((end-start)/60))


if __name__ == "__main__":
    n = int(sys.argv[1])
    algorithm = str(sys.argv[2])

    main(n, algorithm)








