import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
pd.set_option('display.max_columns', None)

#%%

#load data and survey
data1 = pd.read_csv("~\Desktop\cerealDBData\cerealDBSalesDataByProduct.csv")

data1 = data1.rename(columns = {"UPC" : "upc", "DESCRIP" : "cerealName"})

print(data1.head())
print(data1.shape)
print(data1.dtypes)
print(data1.describe())

#%%

profitDirection = []

for i in data1["averageProfit"]:
    if i < 0:
        profitDirection.append("Loss")
    else:
        profitDirection.append("Gain")
data1["profitDirection"] = profitDirection
print(data1.head())
print(data1.shape)
print(data1["profitDirection"].value_counts())

#%%

#Scatterplot build

minSize = min(data1['averageProfit'])
maxSize = max(data1['averageProfit'])

minSize = int(minSize)*-1
maxSize = int(maxSize)*5
#%%

sns.set_style("darkgrid")
plt.figure(figsize=(8, 6),  dpi = 600) 
plot = sns.scatterplot(data=data1, x="unitsSold", y="grossRevenueByProduct", 
                hue = "averageProfit", 
                palette="rocket_r",
                sizes = (minSize, maxSize), 
                alpha = 0.75)

plot.set_title("Cereal Sales Data")
plot.set_xlabel("Units Sold (Millions)")
plot.set_ylabel("Gross Revenue by Product (Tens of Millions)")
plot.legend(title='Average Gross Profit (%)')

#%%

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
    
#%%

sampleBounds (data1['averageProfit'], 95)

#%%

print(data1.sort_values(by = ["averageProfit"], ascending = True))