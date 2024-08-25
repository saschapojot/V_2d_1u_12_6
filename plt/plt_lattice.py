import numpy as np
import glob
import sys
import re
import matplotlib.pyplot as plt
from datetime import datetime
import json
import pandas as pd


#This script loads csv data and plot lattice

if (len(sys.argv)!=2):
    print("wrong number of arguments")
    exit()
rowNum=0#int(sys.argv[1])
unitCellNum=int(sys.argv[1])

csvDataFolderRoot="../dataAll/dataAllUnitCell"+str(unitCellNum)+"/row"+str(rowNum)+"/csvOutAll/"
inCsvFile="../V_inv_12_6Params.csv"

TVals=[]
TFileNames=[]

for TFile in glob.glob(csvDataFolderRoot+"/T*"):

    matchT=re.search(r"T(\d+(\.\d+)?)",TFile)
    # if float(matchT.group(1))<1:
    #     continue

    if matchT:
        TFileNames.append(TFile)
        TVals.append(float(matchT.group(1)))



sortedInds=np.argsort(TVals)
sortedTVals=[TVals[ind] for ind in sortedInds]
sortedTFiles=[TFileNames[ind] for ind in sortedInds]

def pltU_dist(oneTFile):
    """

    :param oneTFile: corresponds to one temperature
    :return: U plots, U mean, U var, dist plots, dist mean, dist var
    """
    matchT=re.search(r'T([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?)',oneTFile)
    TVal=float(matchT.group(1))

    U_distPath=oneTFile+"/U_dist/U_distData.csv"
    df=pd.read_csv(U_distPath)

    filtered_df = df[(df['U'] > -1000) & (df['U'] < 100)]

    UVec=np.array(filtered_df.iloc[:,0])


    print("T="+str(TVal)+", data num="+str(len(UVec)))

    #U part
    meanU=np.mean(UVec)
    meanU2=meanU**2

    varU=np.var(UVec,ddof=1)
    sigmaU=np.sqrt(varU)
    UConfHalfLength=np.sqrt(varU/len(UVec))
    nbins=100
    fig=plt.figure()
    axU=fig.add_subplot()
    (n0,_,_)=axU.hist(UVec,bins=nbins)

    meanUStr=str(np.round(meanU,4))
    print("T="+str(TVal)+", E(U)="+meanUStr)
    sigmaUStr=str(np.round(sigmaU,4))

    axU.set_title("T="+str(TVal))
    axU.set_xlabel("$U$")
    axU.set_ylabel("#")
    xPosUText=(np.max(UVec)-np.min(UVec))*1/2+np.min(UVec)
    yPosUText=np.max(n0)*2/3
    axU.text(xPosUText,yPosUText,"mean="+meanUStr+"\nsd="+sigmaUStr)
    plt.axvline(x=meanU,color="red",label="mean")
    axU.text(meanU*1.1,0.5*np.max(n0),str(meanU)+"$\\pm$"+str(sigmaU),color="red")
    axU.hlines(y=0,xmin=meanU-sigmaU,xmax=meanU+sigmaU,color="green",linewidth=15)

    plt.legend(loc="best")

    EHistOut="T"+str(TVal)+"UHist.png"
    plt.savefig(oneTFile+"/"+EHistOut)

    plt.close()

    ### test normal distribution for mean U
    #block mean
    USelectedAll=UVec

    def meanPerBlock(length):
        blockNum=int(np.floor(len(USelectedAll)/length))
        UMeanBlock=[]
        for blkNum in range(0,blockNum):
            blkU=USelectedAll[blkNum*length:(blkNum+1)*length]
            UMeanBlock.append(np.mean(blkU))
        return UMeanBlock

    fig=plt.figure(figsize=(20,20))
    fig.tight_layout(pad=5.0)
    lengthVals=[2,5,7,10]
    for i in range(0,len(lengthVals)):
        l=lengthVals[i]
        UMeanBlk=meanPerBlock(l)
        ax=fig.add_subplot(2,2,i+1)
        (n,_,_)=ax.hist(UMeanBlk,bins=100,color="aqua")
        xPosTextBlk=(np.max(UMeanBlk)-np.min(UMeanBlk))*1/7+np.min(UMeanBlk)
        yPosTextBlk=np.max(n)*3/4
        meanTmp=np.mean(UMeanBlk)
        meanTmp=np.round(meanTmp,3)
        sdTmp=np.sqrt(np.var(UMeanBlk))
        sdTmp=np.round(sdTmp,3)
        ax.set_title("Bin Length="+str(l))
        ax.text(xPosTextBlk,yPosTextBlk,"mean="+str(meanTmp)+", sd="+str(sdTmp))
    fig.suptitle("T="+str(TVal))
    plt.savefig(oneTFile+"/T"+str(TVal)+"UBlk.png")
    plt.close()


    x00Array=np.array(filtered_df["x00"])
    x01Array=np.array(filtered_df["x01"])
    x10Array=np.array(filtered_df["x10"])
    x11Array=np.array(filtered_df["x11"])

    y00Array=np.array(filtered_df["y00"])
    y01Array=np.array(filtered_df["y01"])
    y10Array=np.array(filtered_df["y10"])
    y11Array=np.array(filtered_df["y11"])

    lattice_x00Array=x00Array-x00Array
    lattice_x01Array=x01Array-x00Array
    lattice_x10Array=x10Array-x00Array
    lattice_x11Array=x11Array-x00Array

    lattice_y00Array=y00Array-y00Array
    lattice_y01Array=y01Array-y00Array
    lattice_y10Array=y10Array-y00Array
    lattice_y11Array=y11Array-y00Array


    x00_avg=np.mean(lattice_x00Array)
    x01_avg=np.mean(lattice_x01Array)
    x10_avg=np.mean(lattice_x10Array)
    x11_avg=np.mean(lattice_x11Array)

    y00_avg=np.mean(lattice_y00Array)
    y01_avg=np.mean(lattice_y01Array)
    y10_avg=np.mean(lattice_y10Array)
    y11_avg=np.mean(lattice_y11Array)



    plt.figure()
    plt.scatter(x00_avg,y00_avg,color="black",label="A00")
    plt.text(x00_avg *1.4, y00_avg +y11_avg*0.1, "A00", fontsize=9, color="black")

    plt.scatter(x01_avg,y01_avg,color="black",label="A01")
    plt.text(x01_avg *0.9, y00_avg +y11_avg*0.1, "A01", fontsize=9, color="black")

    plt.scatter(x10_avg,y10_avg,color="black",label="A10")
    plt.text(x10_avg *0.9, y10_avg*0.9, "A10", fontsize=9, color="black")


    plt.scatter(x11_avg,y11_avg,color="black",label="A11")
    plt.text(x11_avg *0.9, y11_avg*0.9, "A11", fontsize=9, color="black")

    plt.legend(loc="best")
    plt.savefig(oneTFile+"/T"+str(TVal)+"lattice.png")
    plt.close()

    print("x01_avg="+str(x01_avg))
    print("x01_avg/2="+str(x01_avg/2))
    print("x10_avg="+str(x10_avg))



tStatsStart=datetime.now()
for k in range(0,len(sortedTFiles)):
    oneTFile=sortedTFiles[k]
    pltU_dist(oneTFile)