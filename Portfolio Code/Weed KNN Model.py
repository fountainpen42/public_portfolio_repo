import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.tools.eval_measures import rmse
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import silhouette_samples
from sklearn.decomposition import PCA
from matplotlib import colors as mc
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
#%%

#Import Data

data = pd.read_csv("~\Downloads\weed.csv")

print(data.head())
print(data.shape)
print(data.dtypes)

print(data.describe())

#%%

#Data Cleaning:
#We want to ultimately classify weed strains by their THC content along with qualitative feelings from smoking the strain
#So we will drop columns such as tinnitus, muscle fibrosis, and other columns related to why a respondent said they used the strain

data = data.drop(columns = ["ptsd",
                            "bipolar_disorder",
                            "cancer",
                            "cramps",
                            "gastrointestinal_disorder",
                            "inflammation",
                            "muscle_spasms",
                            "eye_pressure",
                            "migraines",
                            "asthma",
                            "anorexia",
                            "arthritis",
                            "add/adhd",
                            "muscular_dystrophy",
                            "hypertension",
                            "glaucoma",
                            "pms",
                            "seizures",
                            "spasticity",
                            "spinal_cord_injury",
                            "crohns_disease",
                            "phantom_limb_pain",
                            "multiple_sclerosis",
                            "parkinsons",
                            "tourettes_syndrome",
                            "alzheimers",
                            "hiv/aids",
                            "tinnitus",
                            "epilepsy",
                            "fibromyalgia"])


data = data.dropna(axis = 0)

#%%

#Check cleaned data

print(data.head())
print(data.shape)
print(data.describe())

for col in data.columns:
    print(col)
    
#%%

#Get rid of other irrelevant columns

data = data.drop(columns = ["Unnamed: 0", "most_common_terpene"])

#%%

print(data.head())
print(data.shape)
print(data.describe())
for col in data.columns:
    print(col)

#%%

#Now, before we can run classification, we need to do dimensional reduction
#For THC level, this variable tupe seems fine as is
#But for the subjective respondent markers, there is likely problems of multicollinearity, plus the curse of dimensionality
#So this seems a great use of PCA

#We will handle the "sensations" data columns seperately from name, type, and thc level

sensationData = data.iloc[:, 3:]

print(sensationData.head())
print(sensationData.describe())

#%%

#Check Correlation
#NOTE: DELETE NAMES FROM PLOT LATER, TOO MANY TO FIT

dataCorr = sensationData.corr(method = "spearman")
print("Spearman Correlation Matrix: ", + dataCorr)

ax = sns.heatmap(dataCorr, 
                 cmap = "YlGnBu_r", 
                 vmin=0.5, vmax=1.0,
                 label='big')
ax.set_title('Spearman Correlation Matrix', fontsize = 24, pad= 30)

#Okay, clearly we have too much correlation, we MUST do dimensional reduction here, everything has a moderate correlation with one another

#%%

#Z-Score normalization & getting matrices from PCA

zScoreData = stats.zscore(sensationData)
pca = PCA()
pca.fit(zScoreData)
eigVals = pca.explained_variance_
loading = pca.components_
rotatedData = pca.fit_transform(zScoreData, 1)

#%%

#Eigenvalues of each PC

covarExplained = eigVals/sum(eigVals)*100
print(eigVals)
print(covarExplained)

#%%

#Generic Screeplot

numColumns = 28
plt.bar(np.linspace(1,numColumns,numColumns),eigVals,color='#3C69F0')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.sans-serif"] = ["Verdana"]
plt.title("Screeplot", pad = 30, fontsize = 24)
plt.xlabel('Principal component', fontsize = 14)
plt.ylabel('Eigenvalue', fontsize = 14)

#%%

#Horns Criteria

nDraws = 10000 # How many repetitions per resampling?
numRows = 2185 # How many rows to recreate the dimensionality of the original data?
numColumns = 28 # How many columns to recreate the dimensionality of the original data?
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


#From this noise distribution/Horn's Criteria analysis, we see that the first 5 PC's are meaningful enough to rise above the noise
#These will be our new variables
#Now let's figure out what they are


#%%

#Loading Matrix analysis and Eigenvector analysis

whichPC = 0
#Generate loading matrix of specified PC in terms of its EVEs, y = PC slice of loading matrix
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.sans-serif"] = ["Verdana"]
plt.bar(np.linspace(1,28,28),loading[:,whichPC], color = "#DE7066")
plt.title("Loading Matrix Plot of Principal Component 1", pad = 30, fontsize = 24)
plt.xlabel('Factor', fontsize = 14)
plt.ylabel('Eigenvector', fontsize = 14)

#Results:
#PC 1- points towards relaxed, and dry_mouth and AWAY from aroused
#PC 2- points towards Lack of Appetite HEAVIlY, energetic, and relaxed
#PC 3- points twoards Focus SUPER HEAVILY, dry_eyes, and giggly
#PC 4- points towards headaches and happy, super AWAY from lack of appetite
#PC 5- points more twords arousal, nausea, giggly, energetic, and creative

#%%

#Reference of columns

for col in sensationData.columns:
    print(col)
    
#%%

#Determine new variables

#PC1 = Bliss?
#PC2 = Excitement?
#PC3 = Flow/Zen
    #This makes a lot of sense, you're super focus so much you forget to blink, you're in a state of flow
#PC4 = Painful Fulfillment?
#PC5 = Euphoria/Mania 
    #This one is also an easy one, you're aroused, nauseous, giggly, energetic, etc. You're in a state of mania

#%%

#Check out new data

newArray = rotatedData[:, :5]

newSensationData = pd.DataFrame(newArray, columns =["bliss", "excitement", "flow", "fulfillment", "mania"])

print(newSensationData.head())
print(newSensationData.shape)
print(newSensationData.describe())

#%%

#Create new dataframe with transformed data + original data's names, types, + thc content

otherData = data[["name", "type", "thc_level"]]
weedData = otherData.join(newSensationData, lsuffix = "redundant")

#%%

print(weedData.head())
print(weedData.describe())
print(weedData.dtypes)

print(weedData.sort_values(by = ["mania"], ascending = False))

#%%

dataCorr = weedData.corr(method = "pearson")
print("Spearman Correlation Matrix: ", + dataCorr)

ax = sns.heatmap(dataCorr, 
                 cmap = "YlGnBu_r", 
                 vmin=0.5, vmax=1.0,
                 label='big')
ax.set_title('Spearman Correlation Matrix', fontsize = 24, pad= 30)

#%%

#KNN model
#Now we can finally use a classification algo to classify strains into hybrid, indica, or sativa using thc content, + sensation data
weedData = weedData.dropna(axis = 0)


weedDataX = weedData[["thc_level",
                   "bliss",
                   "excitement",
                   "flow",
                   "fulfillment",
                   "mania"]]
weedDataY = weedData["type"]
#hotCode = []

#for j in weedDataY:
#    if j == "Indica":
#        hotCode.append(0)
#    elif j == "Sativa":
#        hotCode.append(1)
#    elif j == "Hybrid":
#        hotCode.append(2)

#weedDataY = hotCode

#DONT FORGET TO NORMALIZE DATA OTHERWISE KNN GETS TOO CONFUSED
weedDataXNorm = stats.zscore(weedDataX)
weedDataXNorm = pd.DataFrame(weedDataXNorm, columns =["thc_level", "bliss", "excitement", "flow", "fulfillment", "mania"])

#split data into test and training sets, using test size of 30%
xTrain, xTest, yTrain, yTest = train_test_split(weedDataX, weedDataY, test_size = 0.3, random_state = 4, stratify=weedData['type'])


#%%

#KNN model for any given K

knn = KNeighborsClassifier(n_neighbors = 17)
knn.fit(xTrain, yTrain)
yPred = knn.predict(xTest)

print(accuracy_score(yTest, yPred))
print(classification_report(yTest, yPred))
print(confusion_matrix(yTest, yPred))

#%%

#Finding optimal K based on AUC score, max AUC = better!
#Must use over multi_class for 
scoresList = []
scores = {}
kRange = range(1, 41)

for k in kRange:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(xTrain, yTrain)
    yPred = knn.predict_proba(xTest)
    scores[k] = roc_auc_score(yTest, yPred, average = "weighted", multi_class='ovr')
    scoresList.append(roc_auc_score(yTest, yPred, average = "weighted", multi_class='ovr'))

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams["font.sans-serif"] = ["Verdana"]
plt.title("AUC Scores Across Various K Values", pad = 30, fontsize = 24)
plt.xlabel('K', fontsize = 10)
plt.ylabel('AUC Score (Weighted, Over Multi-Class)', fontsize = 10)
plt.plot(kRange, scoresList)

#%%

#Sort scores in ascending order

print(dict(sorted(scores.items(), key=lambda item: item[1])))

