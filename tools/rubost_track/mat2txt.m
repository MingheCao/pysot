addpath '/home/rislab/Downloads/ECO_raw_results/ECO-HC/OTBr'

imagefiles = dir('./OTBr/*.mat');      
nfiles = length(imagefiles); 

for i = 1: nfiles
    nn=imagefiles(i).name;
    m=load(nn);
    rects=m.results{1,1}.res;
    
    path=['/home/rislab/Downloads/ECO_raw_results/ECO-HC/','UAV123/',erase(nn,'mat'),'txt'];

    writematrix(rects,path);
    
end
    