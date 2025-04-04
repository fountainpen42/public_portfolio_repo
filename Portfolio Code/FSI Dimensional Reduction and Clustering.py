import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors as mc
import glob, os
from numpy import array, linspace
from sklearn.neighbors.kde import KernelDensity
from matplotlib.pyplot import plot
from scipy.signal import argrelextrema
pd.set_option('display.max_columns', 5)
#%%

#Load and glob data

path = r'~\\Desktop\Fragile State Index Data'
all_files = glob.glob(os.path.join(path, "*.csv")) 
df_from_each_file = (pd.read_csv(f) for f in all_files)
data = pd.concat(df_from_each_file)
print(data[["Country", "P3: Human Rights"]])
#%%

#Clean data, take median of scores since data is multiyear
#Higher scores on any measure means more unstable

data2 = data.drop(columns = ["Rank", "Year", "Change from Previous Year"])
data2 = data2.groupby(["Country"]).median()
data2 = data2.reset_index()
print(data2)
print(data2.shape)
PCAdata = data2.drop(columns = ["Country", "Total"])
PCAdata = PCAdata.dropna(axis = 0)

corrCleanData = PCAdata.rename(columns = {"C1: Security Apparatus" : "C1",
                                          "C2: Factionalized Elites" : "C2",
                                          "C3: Group Grievance" : "C3",
                                          "E1: Economy" : "E1",
                                          "E2: Economic Inequality" : "E2",
                                          "E3: Human Flight and Brain Drain" : "E3",
                                          "P1: State Legitimacy" : "P1",
                                          "P2: Public Services" : "P2",
                                         "P3: Human Rights" : "P3",
                                         "S1: Demographic Pressures" : "S1",
                                         "S2: Refugees and IDPs" : "S2",
                                         "X1: External Intervention" : "X1"})

#%%
#Check multicollinearity
dataCorr = corrCleanData.corr(method = "pearson")
print("Pearson Correlation Matrix: ", + dataCorr)

#cmap = mc.ListedColormap(["#283C3E", "#4B6A6C", "#618F94", "#709599", "#86B3B8", "#A9DADE"])
sns.set(font = "Helvetica Neue", font_scale=1.0)
ax = sns.heatmap(dataCorr, 
                 cmap = "YlGnBu_r", 
                 vmin=0.5, vmax=1.0,
                 label='big')
ax.set_title('Pearson Correlation Matrix', fontsize = 24, pad= 30)

#%%

#Fit PCA and get relevant stats

zScoreData = stats.zscore(PCAdata)
pca = PCA()
pca.fit(zScoreData)
eigVals = pca.explained_variance_
loading = pca.components_
rotatedData = pca.fit_transform(zScoreData, 1)
#%%

#Kaiser and 90% Criteria

covarExplained = eigVals/sum(eigVals)*100
print(eigVals)
print(covarExplained)
#%%

#Horns Criteria

nDraws = 10000 #repetitions per resampling
numRows = 181 # How many rows from original dataset
numColumns = 12 # How many columns from original dataset
eigSata = np.empty([nDraws,numColumns]) # Initialize array to keep eigenvalues of sata
eigSata[:] = np.NaN # Convert to NaN

for i in range(nDraws):
    # Draw the sata from a normal distribution:
    sata = np.random.normal(0,1,[numRows,numColumns]) 
    pca = PCA()
    pca.fit(sata)
    temp = pca.explained_variance_
    eigSata[i] = temp
    
plt.bar(np.linspace(1,numColumns,numColumns),eigVals,color='#3C69F0')
plt.plot(np.linspace(1,numColumns,numColumns),np.transpose(eigSata),color='#53F399')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.sans-serif"] = ["Verdana"]

plt.title("Screeplot + Noise Distribution", pad = 40, fontsize = 24)
plt.suptitle("Horn's Criteria", fontsize = 18)
plt.xlabel('Principal Component (SATA)', fontsize = 14)
plt.ylabel('Eigenvalue of SATA', fontsize = 14)
plt.legend(['SATA'])
#%%

#Scree Plot + Kaiser Line
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.sans-serif"] = ["Verdana"]
numColumns = 12
plt.bar(np.linspace(1,numColumns,numColumns),eigVals,color='#3C69F0')
plt.plot([1,numColumns],[1,1],color='#9302BF')
plt.title("Screeplot + Kaiser Line", pad = 40, fontsize = 24)
plt.suptitle("Kaiser Criteria", fontsize = 18)
plt.xlabel('Principal component', fontsize = 14)
plt.ylabel('Eigenvalue', fontsize = 14)
#%%

#Generic Scree Plot
numColumns = 12
plt.bar(np.linspace(1,numColumns,numColumns),eigVals,color='#3C69F0')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.sans-serif"] = ["Verdana"]
plt.title("Screeplot", pad = 30, fontsize = 24)
plt.xlabel('Principal component', fontsize = 14)
plt.ylabel('Eigenvalue', fontsize = 14)

#%%

#Loading Matrix Analysis

whichPC = 0
#Generate loading matrix of specified PC in terms of its EVEs, y = PC slice of loading matrix
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.sans-serif"] = ["Verdana"]
plt.bar(np.linspace(1,12,12),loading[:,whichPC], color = "#DE7066")
plt.title("Loading Matrix Plot of Principal Component 1", pad = 30, fontsize = 24)
plt.xlabel('Factor', fontsize = 14)
plt.ylabel('Eigenvector', fontsize = 14)

#%%

#Create new Dataframe with the transformed values and align them with their respective countries

newData = pd.DataFrame(rotatedData, columns = ["Humanitarian Public Goods", "B", "C", "D", "E", "F", "B", "C", "D", "E", "F", "B"])
newData = data2.join(newData, lsuffix = "redundant")
newData = newData[["Country", "Humanitarian Public Goods"]]
print(newData)

print(newData.sort_values(by = "Humanitarian Public Goods", ascending = False))
#plt.hist(x = newData["Principal Component"], bins = 15)

#%%

#Histogram and stats

plt.hist(newData["Humanitarian Public Goods"], bins = 20, color = "#53F399")
plt.axvline(newData["Humanitarian Public Goods"].median(), color='#9302BF', linestyle='solid', linewidth=2)
min_ylim, max_ylim = plt.ylim()
plt.text(newData["Humanitarian Public Goods"].median()*0.2, max_ylim*0.8, 'Median: {:.2f}'.format((newData["Humanitarian Public Goods"].median())))
plt.title("Histogram of HPG Distribution", pad = 30, fontsize = 24)
plt.xlabel('Humanitarian Public Goods', fontsize = 14)
plt.ylabel('Count', fontsize = 14)
print(newData["Humanitarian Public Goods"].mean())
print(newData["Humanitarian Public Goods"].std())
#%%

#Kernel Density Estimation

#Create new array which just contains the values of the HPG, this is a 1D array
newDataHPG = newData.to_numpy()
newDataHPG = newDataHPG[:, 1].reshape(-1, 1)

#Calculate optimal bandwith using Silverman's rule
#Bandwith is essentially how large of a window do you integrate over to calculate the probability at the center of the window
#A larger bandiwth means more points are considered and the resultant distribution is smoother but less descriptive since more points are considered, vice versa for smaller ones
#As a result, picking the best one is a matter of bias vs variance, we don't want to underfit the model with a large bandwith nor overfit with a low bandwith
#Silverman's rule is a formula to do so by reducing the mean integrated square error, typically for gaussian kernels
#181 refers to the sample size, all other numbers here are constants
m = min((np.var(newDataHPG))**(1/2), ((stats.iqr(newDataHPG))/1.349))
h = ((0.9 * m) / (181 ** (1/5)))
print(h)


#Perform the KDE
kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(newDataHPG)
s = linspace(min(newDataHPG),max(newDataHPG))
e = np.exp(kde.score_samples(s.reshape(-1,1)))
#Plot the probability density distribution
fig = plt.figure(figsize=(8,5))
plt.title("Probability Density Distribution of HPG", pad = 30, fontsize = 24)
plt.xlabel('Humanitarian Public Goods', fontsize = 14)
plt.ylabel('Probability', fontsize = 14)
plt.plot(s, e, color = "#3C69F0")
plt.show()
#%%

#Find local Maxima and Minima based on derivation
#Local Minima act as the interval cut offs or cut offs for the "clusters"
#Local Maxima act as the centroid or "cluster center" for each interval
mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
print ("Minima:", + s[mi])
print ("Maxima:", + s[ma])

plot(s, e, color = "#3C69F0")
#color minima, maxima, and intervals
plot(s[:mi[0]+1], e[:mi[0]+1], 'r',
     s[mi[0]:mi[0]], 'b',
     s[ma], e[ma], 'bo',
     s[mi], e[mi], 'go')

plt.title("Probability Density Distribution of HPG", pad = 30, fontsize = 24)
plt.xlabel('Humanitarian Public Goods', fontsize = 14)
plt.ylabel('Probability', fontsize = 14)

#%%

#Sample Bounds

def sampleBounds(inputArray, bound):
    #calculate mass for both tails
    tailMass = (100-bound)/2
    lowerQuant = tailMass / 100
    upperQuant = (bound + tailMass) / 100
    upperBound = np.quantile(inputArray, upperQuant, interpolation = "lower")
    lowerBound = np.quantile(inputArray, lowerQuant, interpolation = "lower")
    
    print("Lower Bound: ", + lowerBound)
    print("Upper Bound: ", + upperBound)
    
sampleBounds(newDataHPG, 90)


