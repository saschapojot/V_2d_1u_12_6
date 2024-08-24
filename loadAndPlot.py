import numpy as np

import glob
from decimal import Decimal
import pickle
import re
import matplotlib.pyplot as plt
import pandas as pd
import sys

from mk_dir import dataRoot


#This script loads  and plots the data in the last few files
def format_using_decimal(value):
    # Convert the float to a Decimal
    decimal_value = Decimal(value)
    # Remove trailing zeros and ensure fixed-point notation
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)

if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()

T=float(sys.argv[1])
unitCellNum=int(sys.argv[2])

TStr=format_using_decimal(T)

inParamFileName="./V_inv_12_6Params.csv"
rowNum=0
inDf=pd.read_csv(inParamFileName)
oneRow=inDf.iloc[rowNum,:]
a1=float(oneRow.loc["a1"])
b1=float(oneRow.loc["b1"])


def V1(r):
    return a1*r**(-12)-b1*r**(-6)

def sort_data_files_by_swEnd(oneDataFolder):
    """

    :param oneDataFolder:
    :return:
    """


    dataFolderName=oneDataFolder
    dataFilesAll=[]
    sweepEndAll=[]

    for oneDataFile in glob.glob(dataFolderName+"/*.pkl"):
        dataFilesAll.append(oneDataFile)
        matchEnd=re.search(r"sweepEnd(\d+)",oneDataFile)
        if matchEnd:
            sweepEndAll.append(int(matchEnd.group(1)))


    endInds=np.argsort(sweepEndAll)
    # sweepStartSorted=[sweepStartAll[i] for i in startInds]
    sortedDataFiles=[dataFilesAll[i] for i in endInds]

    return sortedDataFiles

dataPath=dataRoot+"/dataAllUnitCell"+str(unitCellNum)+"/row0/T"+TStr+"/U_dist_dataFiles/"


plt_nameU="U"

inUPath=dataPath+"/"+plt_nameU+"/"

sorted_inUFiles=sort_data_files_by_swEnd(inUPath)

lastFilesNum=10

files2Plot=sorted_inUFiles[-lastFilesNum:]

arrU=np.array([])
for pkl_file in files2Plot:
    with open(pkl_file,"rb") as fptr:
        arrIn=pickle.load(fptr)
        arrU=np.append(arrU,arrIn)

avg_arrU=arrU/unitCellNum

avg_arrU=avg_arrU[avg_arrU<100]
avg_arrU=avg_arrU[avg_arrU>-1000]

plt.figure(figsize=(120,20))
plt.scatter(range(0,len(avg_arrU)),avg_arrU,s=1)
plt.title("T="+str(TStr)+", N="+str(unitCellNum)+", avg U in last "+str(lastFilesNum)+" files")
plt.savefig("T"+TStr+"N"+str(unitCellNum)+"lastFilesU.png")
plt.close()
# peaksU, _ = find_peaks(arrU)
# periods = np.diff(peaksU)
# average_period = np.mean(periods)
#
# # Output the results
# print(f"Periods between peaks: {periods}")
# print(f"Average period: {average_period}")
#
# print(np.sqrt(np.var(periods,ddof=1)))




def autocorrelation(x, lag):
    x_mean = np.mean(x)
    x_var = np.var(x)
    N = len(x)

    # Compute autocorrelation for given lag
    acf = np.sum((x[:N-lag] - x_mean) * (x[lag:] - x_mean)) / ((N-lag) * x_var)

    return acf

# autcU=autocorrelation(arrU,100000)
# print(autcU)
