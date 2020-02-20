# K-modes-analysis
The K-means algorithm works only with interval features.  One way to apply the k-means algorithm to categorical features is to transform them into a new interval feature space. However, this approach can be very inefficient, and it does not produce good results.
For clustering categorical features, we should consider the K-modes clustering algorithm which extends the K-means algorithm by using different dissimilarity measures and a different method for computing cluster centers.  See this article for more details. Huang, Z. (1997). “A Fast Clustering Algorithm to Cluster Very Large Categorical Data Sets in Data Mining.” In Proceedings of the SIGMOD Workshop on Research Issues on Data Mining and Knowledge Discovery, 1–8. New York: ACM Press.
Please implement the K-modes clustering method in Python and then apply the method to the cars.csv.  Your input fields are these four categorical features: Type, Origin, DriveTrain, and Cylinders.  Please do not remove the missing or blank values in these four features.  Instead, consider these values as a separate category.
The cluster centroids are the modes of the input fields.  In the case of tied modes, choose the lexically or numerically lowest one.
Suppose a categorical feature has observed values v_1,…,v_p.  Their frequencies (i.e., number of observations) are f_1,…,f_p.  The distance metric between two values is d(v_i,v_j )=0 if v_i= v_j.  Otherwise, d(v_i,v_j )=1/f_i +1/f_j .  The distance between any two observations is the sum of the distance metric of the four categorical features.
	(5 points) What are the frequencies of the categorical feature Type?

	(5 points) What are the frequencies of the categorical feature DriveTrain?

	(5 points) What is the distance between Origin = ‘Asia’ and Origin = ‘Europe’?

	(5 points) What is the distance between Cylinders = 5 and Cylinders = Missing?

	(5 points) Apply the K-modes method with three clusters.  How many observations in each of these three clusters?  What are the centroids of these three clusters?

	(5 points) Display the frequency distribution table of the Origin feature in each cluster.
