import glob 
import os

files=glob.glob('/home/rislab/Downloads/ECO_raw_results/ECO-HC/UAV123/*.mat')

for f in files:
    name = f.split('/')[-1].replace('OPE_uav_','')
    name= '/'.join(f.split('/')[:-2]) + '/OTBr/' + name
    os.rename(f,name)
