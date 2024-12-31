import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors as mc
import glob, os
pd.set_option('display.max_columns', 5)
#%%

#Load up the data

path = r'C:\Users\Fountainpen\Desktop\Fragile State Index Data' #sets file path to folder containing all relevant csv files
all_files = glob.glob(os.path.join(path, "*.csv")) #glob finds all pathnames matching the given pattern, so it "looks" inside folder
df_from_each_file = (pd.read_csv(f) for f in all_files) #makes a list of all the files found from the globbing
data = pd.concat(df_from_each_file) #pandas appends ach csv onto one another in a series and creates a new dataframe with all the found csvs
print(data[["Country", "P3: Human Rights"]])
#%%

#Note that the data is long form for the variable "year," as in the same country will show up as an observation with a different year and measure
#Ok, on a second thought, I MUST readjust this data, if I put it in raw, it will ultimately end up being pseudoreplication and temporaly auto-correlated
#My Unit of Analaysis is COUTNRY, NOT COUNTRY YEAR!
#Because of this, a country's X year measurement PROBABLY PREDICTS THE NEXT YEAR'S MEASUREMENT: this is temporaly autocorrelated!!! 
#The information for the countries is not truly independent in this form!

#%%

#To Regularlize the unit of analysis, I will take the median for each country across all indicator rows + total: this is ORDINAL DATA
#Higher scores on any measure means MORE UNSTABLE

data2 = data.drop(columns = ["Rank", "Year", "Change from Previous Year"])
data2 = data2.groupby(["Country"]).median()
#Reset index to make country a column
data2 = data2.reset_index()
print(data2)
print(data2.shape)
#Now we have truly clean data that should not be temporaly auto-correlated
#%%

#Actual PCA

#create new data set containing only IVs/Factors for PCA
PCAdata = data2.drop(columns = ["Country", "Total"])
#clean up data, ready to go
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

#Check multicollinearity: Graphed with a heatmap
dataCorr = corrCleanData.corr(method = "pearson")
print("Pearson Correlation Matrix: ", + dataCorr)

#cmap = mc.ListedColormap(["#283C3E", "#4B6A6C", "#618F94", "#709599", "#86B3B8", "#A9DADE"])
sns.set(font = "Helvetica Neue", font_scale=1.0)
ax = sns.heatmap(dataCorr, 
                 cmap = "YlGnBu_r", 
                 vmin=0.5, vmax=1.0,
                 label='big')
ax.set_title('Pearson Correlation Matrix', fontsize = 24, pad= 30)
#Results show that many of the variables are HIGHLY correlated with one another. It'd be foolish not to perform some kind of method to deal with this issue

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
#Do I know how the calculations are made in particular? No
#Do I know how to interpret it? Yup

nDraws = 10000 # How many repetitions per resampling?
numRows = 181 # How many rows to recreate the dimensionality of the original data?
numColumns = 12 # How many columns to recreate the dimensionality of the original data?
eigSata = np.empty([nDraws,numColumns]) # Initialize array to keep eigenvalues of sata
eigSata[:] = np.NaN # Convert to NaN

for i in range(nDraws):
    # Draw the sata from a normal distribution:
    sata = np.random.normal(0,1,[numRows,numColumns]) 
    # Run the PCA on the sata:
    pca = PCA()
    pca.fit(sata)
    # Keep the eigenvalues:
    temp = pca.explained_variance_
    eigSata[i] = temp
    
plt.bar(np.linspace(1,numColumns,numColumns),eigVals,color='#3C69F0') # plot eig_vals from section 4
plt.plot(np.linspace(1,numColumns,numColumns),np.transpose(eigSata),color='#53F399') # plot eig_sata
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

#Ok, by the Kaiser, Elbow, and Horns' Criterias, there is only ONE meaningful principal component here that explains enough of the variance in the data
#This is rather suprising, I'd expect at least 2 or 3 more meaningul components
#But it appears that the manifold the PCA is describing is rather low dimensional
#There could be an argument that the 90% criteria might be the more meaningful one here though... since countries are far more complex
#But wasn't that the point of the PCA in the first place?

#%%

#Loading Matrix Analysis

#Remember: 1st we index from 0, 0 = PC number 1
#Postive vectors' underlying factor implies a positive correlation to this PC
#Negative vectors' underlying factor contribute negatively and linearly to the value

whichPC = 0
#Generate loading matrix of specified PC in terms of its EVEs, y = PC slice of loading matrix
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.sans-serif"] = ["Verdana"]
plt.bar(np.linspace(1,12,12),loading[:,whichPC], color = "#DE7066")
plt.title("Loading Matrix Plot of Principal Component 1", pad = 30, fontsize = 24)
plt.xlabel('Factor', fontsize = 14)
plt.ylabel('Eigenvector', fontsize = 14)

#%%

#Based on the loading matrix for PC 1, it points strongly and postively twoards metrics 8 and 9
#It points most negatively away from metric 1

#Positive Eigenvectors:
    #8 = (Access to) Public Services
    #9 = Human Rights
    
#Negative Eigenvectors:
    #1 = Security Apparatus

#So now we need to interpret this- I will say this represents Humanitarian Public Goods or Humanitarian Development
    #This appears to describe the descriptive eigenvectors the best, how well can a country provide equitable rule of the law and equitable and avaliable public infrsturcture
    #Are the laws of a country equitable and respect human rights?
    #And are citizens able to equally and readily access the infrastructure of acountry? Like schools and hospitals?
    #This makes sense that it'd cature data pertaining to all the factors.
    #Countries with poor security probably have bad infrastructure to sustain a military and police
    #Coutnries with group grievances and partisanship probably have poor human rights in regards to minority's rights
#%%

#Create new Dataframe with the transformed values and align them with their respective countries

newData = pd.DataFrame(rotatedData, columns = ["Humanitarian Public Goods", "B", "C", "D", "E", "F", "B", "C", "D", "E", "F", "B"])
newData = data2.join(newData, lsuffix = "redundant")
newData = newData[["Country", "Humanitarian Public Goods"]]
print(newData)

print(newData.sort_values(by = "Humanitarian Public Goods", ascending = False))
#as a little sanity check, putting the values in descending order shows it makes sense, Somalia is at dead last and Finland is the most stable countries
#this aligns with one would naturally expect
#plt.hist(x = newData["Principal Component"], bins = 15)

#%%

#Ok so it appears that countries are along an axis of humanitarian development
#Equal and ready access to public goods; along with respect for human rights/equitable justice
#Note that this description of public goods EXCLUDES military protection as a public good! It's not as important as one would think!
#This is a slightly heartening thing to observe, the ability of a state to protect itself from violence with violence does not matter as much as how well the state protects equality and quality of life for all
#Bullets and bombs don't make a more stable society, it appears human rights and infrastructure create stability

#%%

#Here's the main rub though, it doesn't make sense to do linear regression or correlation anymore since we reduced all the IVs to just 1 now
#It would just be a straight line between the total fragility and  which is kind of a no shit sherlock scenario
#since the total fragility is entirely calculated from the indicators, it's not much of a predictor anymore is it now
#We can't do clustering since this is 1-dimensional data now
#Logistic regression makes no sense
#Literally the best we can do is a histogram to see the distribution of the data to describe this dimensionally reduced data set
#but we can take this one step further...

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
#Instead of using a histogram and calling it a day, let's try to get a proper probability distribution here, the histogram clearly shows that this data is not distributed normally or in some other clear fashion!
#Kernel Density Estimation returns the probability density distribution of a random variable, in this case the "Humanitarian Public Goods"
#after all a histogram is just a crude and rough probability density distribution at the end of the day...
#Then we can divi up the distribution based on its local minima to get our "clusters", or in this case intervals
#Local maxima act as the interval centers
#This is a form of integration (calculus)

from numpy import array, linspace
from sklearn.neighbors.kde import KernelDensity
from matplotlib.pyplot import plot
from scipy.signal import argrelextrema

#Create new array which just contains the values of the HPG, this is a 1D array
newDataHPG = newData.to_numpy()
newDataHPG = newDataHPG[:, 1].reshape(-1, 1)

#Calculate optimal bandwith using Silverman's rule
#Bandwith is essentially how large of a window do you integrate over to calculate the probability at the center of the window
#A larger bandiwth means more points are considered and the resultant distribution is smoother but less descriptive since more points are considered, vice versa for smaller ones
#As a result, picking the best one is a matter of bias vs variance, we don't want to underfit the model with a large bandwith nor overfit with a low bandwith
#Silverman's rule is a formula to do so by reducing the mean integrated square error, typically for gaussian kernels
#should be robust
#181 refers to the sample size, all other numbers here are constants
m = min((np.var(newDataHPG))**(1/2), ((stats.iqr(newDataHPG))/1.349))
h = ((0.9 * m) / (181 ** (1/5)))
print(h)


#Perform the KDE
#Here's the gambit:
    #Integrate across the entire array in sliced windows called kernels
    #the shape of the kernel can be some kind of distribution, normal, gaussian, uniform/tophat, etc.
    #the size of the kernel is the bandwith parameter, Silverman's rule seems the most straightforward and a sound way to get the optimal one here
    #Lastly, since KDE retuns logs, take the exp to get the probabilties
kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(newDataHPG)
#the interval of this data set will naturally be the min and max values of the entire array
s = linspace(min(newDataHPG),max(newDataHPG))
#Since KDE uses natural logs to calculate the scores, we need to take the inverse of the function to convert it back into a reasonable scale
#ie. turn the scores back into PROBABILITIES!
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
#being that this is a density distribution let's find the sample bounds to see where the bulk of the data falls into

def sampleBounds(inputArray, bound):
    #calculate mass for BOTH tails
    tailMass = (100-bound)/2
    #convert "tail masses" into appropriate quantiles for upper and lower bounds
    lowerQuant = tailMass / 100
    upperQuant = (bound + tailMass) / 100
    #calculate respective quantiles
    upperBound = np.quantile(inputArray, upperQuant, interpolation = "lower")
    lowerBound = np.quantile(inputArray, lowerQuant, interpolation = "lower")
    
    print("Lower Bound: ", + lowerBound)
    print("Upper Bound: ", + upperBound)
    
sampleBounds(newDataHPG, 90)


