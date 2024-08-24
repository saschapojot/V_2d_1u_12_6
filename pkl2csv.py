import pickle
import numpy as np
import pandas as pd
import statsmodels.api as sm
import warnings
import matplotlib.pyplot as plt
import glob
import re
from decimal import Decimal
import sys
#this script prints part of an array




if (len(sys.argv)!=3):
    print("wrong number of arguments")
    exit()


def format_using_decimal(value):
    # Convert the float to a Decimal
    decimal_value = Decimal(value)
    # Remove trailing zeros and ensure fixed-point notation
    formatted_value = decimal_value.quantize(Decimal(1)) if decimal_value == decimal_value.to_integral() else decimal_value.normalize()
    return str(formatted_value)
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


# cellInd=7
T=float(sys.argv[1])
unitCellNum=int(sys.argv[2])

TStr=format_using_decimal(T)
dataRoot="./dataAll/dataAllUnitCell"+str(unitCellNum)+"/row0/T"+TStr+"/U_dist_dataFiles/"

print(dataRoot)

# print(in_xAPath)


inUPath=dataRoot+"/U/"

sorted_inUFiles=sort_data_files_by_swEnd(inUPath)

# print(sorted_in_xAFiles)
fileInd=-2

inFileU=sorted_inUFiles[fileInd]


# arr=np.reshape(arr,(-1,2*N+1))

inParamFileName="./V_inv_12_6Params.csv"
rowNum=0
inDf=pd.read_csv(inParamFileName)
oneRow=inDf.iloc[rowNum,:]
a1=float(oneRow.loc["a1"])
b1=float(oneRow.loc["b1"])


def V1(r):
    return a1*r**(-12)-b1*r**(-6)
def auto_corrForOneColumn(colVec):
    """

    :param colVec: a vector of data
    :return:
    """
    same=False
    eps=1e-2
    NLags=int(len(colVec)*3/4)
    print("NLags="+str(NLags))
    with warnings.catch_warnings():
        warnings.filterwarnings("error")
    try:
        acfOfVec=sm.tsa.acf(colVec,nlags=NLags)
    except Warning as w:
        same=True
    acfOfVecAbs=np.abs(acfOfVec)
    minAutc=np.min(acfOfVecAbs)
    print("minAutc="+str(minAutc))

    lagVal=-1
    if minAutc<=eps:
        lagVal=np.where(acfOfVecAbs<=eps)[0][0]
    # np.savetxt("autc.txt",acfOfVecAbs[lagVal:],delimiter=',')

    return same,lagVal




print(inFileU)
with open(inFileU,"rb") as fptr:
    UVec=np.array(pickle.load(fptr))

ULength=int(len(UVec))
print("len(UVec)="+str(len(UVec)))
UPart=UVec[-ULength:]/unitCellNum
sameU,lagU=auto_corrForOneColumn(UPart)
print("lagU="+str(lagU))
UDiff=UPart[1:]-UPart[:-1]


plt.figure()
plt.scatter(range(0,len(UPart)),UPart,s=1)
plt.title("UAll")

plt.savefig("UAll.png")
plt.close()
print(UPart)