# Twitter-tweets-clustering-using-K-Means
Script name - k_means.py
Input argument format: "numberOfClusters" "input-file-name" "output-file-name"
Result: Result would be generated as an output file

Example command to Execute:
python k_means.py 5 http://www.utdallas.edu/~axn112530/cs6375/unsupervised/test_data.txt out_file_1.txt
	

RESULT: (for 5 values of K)
---------------------------------------------------------------------------------
Test Run	   Number of Clusters (K)	                   Validation SSE
----------------------------------------------------------------------------------
1.		          	  05			                1.4978420947856068
2.		          	  10			                0.8439711008297258
3.			          20			                0.3468753392857143
4.			          25			                0.2919055611111112
5.			          40			                0.13719377380952383


Note: The values of Validation SSE (Sum of Square Error) differ for each execution, since initial centroids are picked randomly.
Observation: As number of clusters increase, value of SSE decreases. 
