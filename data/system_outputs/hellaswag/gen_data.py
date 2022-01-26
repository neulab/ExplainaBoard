import os
import sys

fin = open("hellaswag.random","w")

for k in range(2000):
    fin.write(str(k) + "\t" + "2" + "\n")

fin.close()