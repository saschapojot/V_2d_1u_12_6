import numpy as np
import matplotlib.pyplot as plt
import pickle



in_pkl_file="./dataAll/dataAllUnitCell1/row0/T1/U_dist_dataFiles/converted_data/latticeFile_sweepEnd99999.pkl"


with open(in_pkl_file,"rb") as fptr:
    arr=pickle.load(fptr)

row=arr[0,:]


x00,y00,x01,y01,x10,y10,x11,y11=row



lattice_x00=x00-x00
lattice_x01=x01-x00
lattice_x10=x10-x00
lattice_x11=x11-x00

lattice_y00=y00-y00
lattice_y01=y01-y00
lattice_y10=y10-y00
lattice_y11=y11-y00


plt.figure()

plt.scatter(lattice_x00,lattice_y00,color="black")
plt.scatter(lattice_x01,lattice_y01,color="black")
plt.text(lattice_x01 *0.9, lattice_y01*0.9, "A01", fontsize=9, color="black")

plt.scatter(lattice_x10,lattice_y10,color="black")
plt.text(lattice_x10 *0.9, lattice_y10*0.9, "A10", fontsize=9, color="black")

plt.scatter(lattice_x11,lattice_y11,color="black")
plt.text(lattice_x11 *0.9, lattice_y11*0.9, "A11", fontsize=9, color="black")


plt.savefig("tmpLattice.png")