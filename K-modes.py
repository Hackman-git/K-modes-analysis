'''
Name: Olugbenga Abdulai
CWID: A20447331
'''

from kmodes.kmodes import KModes
# The kmodes package is from: https://github.com/nicodv/kmodes
import pandas as pd

# reading data
cars = pd.read_csv(r"C:\Users\abdul\Desktop\CS 584\Lectures\Week 4\cars.csv")

# subsetting the data to only required columns
cars = cars[["Type", "Origin", "DriveTrain", "Cylinders"]]

'''
2(a)
'''
# frequencies of the categorical feature Type
print("\nfrequencies of Type attribute:\n", cars.Type.value_counts())

'''
2(b)
'''
# frequencies of the categorical feature DriveTrain
print("\nfrequencies of DriveTrain attribute:\n", cars.DriveTrain.value_counts())

'''
2(c)
'''
freq_origin = cars.Origin.value_counts()
freq_asia = freq_origin.loc['Asia']
freq_europe = freq_origin.loc['Europe']

asia_europe_dist = (freq_asia + freq_europe) / (freq_asia * freq_europe)
print("\nAsia-Europe distance: ", asia_europe_dist)

'''
2(d)
'''
# count of NAs in Cylinders attribute
freq_nans = cars.Cylinders.isna().sum()

# distribution of the other categories
freq_cyl = cars.Cylinders.value_counts()

freq_five_cyl = freq_cyl.iloc[3]
fiveCyl_nan_dist = (freq_five_cyl + freq_nans) / (freq_five_cyl * freq_nans)
print("\nfiveCylinders-NaN distance: ", fiveCyl_nan_dist)

'''
2(e)
'''
# filling NaNs with 'none'
cars.Cylinders = cars.Cylinders.fillna('none')

# ensuring uniform data type across all attributes
cars.Cylinders = cars.Cylinders.astype('str')

# initializing the kmodes algorithm
kmodes = KModes(n_clusters=3, init='Huang')

# fitting and classifying
clust = kmodes.fit_predict(cars)

# obtaining centroids
print("\nCluster centroids:\n", kmodes.cluster_centroids_)

# obtaining count of observations per cluster
print("\ncluster 1 observations: ",list(kmodes.labels_).count(0))
print("cluster 2 observations: ",list(kmodes.labels_).count(1))
print("cluster 3 observations: ",list(kmodes.labels_).count(2))

'''
2(f)
'''
# initializing empty lists for the cluster rows
cluster_1_rows = []
cluster_2_rows = []
cluster_3_rows = []

# obtaining the rows corresponding to each cluster from the labels method
for i in range(len(kmodes.labels_)):
    if  kmodes.labels_[i] == 0:
        cluster_1_rows.append(i)
    elif kmodes.labels_[i] == 1:
        cluster_2_rows.append(i)
    else:
        cluster_3_rows.append(i)

# creating dataframes from the rows in each cluster
cluster_1_df = pd.DataFrame(cars.iloc[cluster_1_rows], columns=cars.columns)
cluster_2_df = pd.DataFrame(cars.iloc[cluster_2_rows], columns=cars.columns)
cluster_3_df = pd.DataFrame(cars.iloc[cluster_3_rows], columns=cars.columns)

# frequency distribution
print("\nCluster 1 origin value distribution:\n", cluster_1_df.Origin.value_counts())
print("\nCluster 2 origin value distribution:\n", cluster_2_df.Origin.value_counts())
print("\nCluster 3 origin value distribution:\n", cluster_3_df.Origin.value_counts())
