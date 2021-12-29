# SAT_algorithms- 

Implementation of https://onlinelibrary.wiley.com/doi/pdf/10.1002/rsa.20057 [1]  




Run the test with:

./rand_planted_test_threads.py $1 $2

$1 is the number of litrals N.

$2 is the algorithm:

  'WP' - Warning Propagation.
  
  'BP' - Belief Propagation.
  
  'SP' - Survey Propagation.

For each kind of SAT problem there is also a comparison with the performance of Majority Vote algorithm.

The test is running for random planted SAT and random SAT problems, but can easily change for CNF files.



[1] Braunstein, Alfredo, Marc Mézard, and Riccardo Zecchina. "Survey propagation: An algorithm for satisfiability." Random Structures & Algorithms 27.2 (2005): 201-226.
