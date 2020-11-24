clc;
clear all;
%% global variable              
global FILE_COUNT;
global TOTALCT;
global CREATED_FLAG;
%% load orginal HS labels

string = 'train';
string2 = 'clean';
if strcmp(string, 'train') == 1

    hyper_dir = './data/train_data/mat/';
    label=dir(fullfile(hyper_dir,'*.mat'));

    
    if strcmp(string2, 'clean') == 1   
        rgb_dir = './data/train_data/rgb_clean/';
    else  
        rgb_dir = './data/train_data/rgb_real/';    
    end
    
     order= randperm(size(label,1));
    
    
else

    hyper_dir = './data/valid/mat/';
    label=dir(fullfile(hyper_dir,'*.mat'));
   
    
    if strcmp(string2, 'clean') == 1   

        rgb_dir = './data/valid/validClean/';    
    else  
         rgb_dir = './data/valid/validReal/';
 
    end
    
     order= randperm(size(label,1));
    
end  
%% Initialization the patch and stride
size_input=50;
size_label=50;
label_dimension=31;
data_dimension=3;
stride=80;


%% Initialization the hdf5 parameters

prefix=[ string '_si50_st80_jpg'];
chunksz=64;
TOTALCT=0;
FILE_COUNT=0;
amount_hd5_image=300;
CREATED_FLAG=false;


%% For loop  RGB-HS-HD5  
for i=1:size(label,1)
     if mod(i,amount_hd5_image)==1     
         filename=Get_filename(label(order(i)).name,prefix,hyper_dir);
     end
    name_label=strcat(hyper_dir,label(order(i)).name); 
    a_temp=struct2cell(load(name_label,'rad'));
    hs_label=cell2mat(a_temp);
    hs_label=hs_label/(2^12-1);

 
    if strcmp(string2, 'clean') == 1   
        rgb_name=[ rgb_dir label(order(i)).name(1:end-4) '_clean.png'];
    else  
        rgb_name=[ rgb_dir label(order(i)).name(1:end-4) '_camera.jpg'];
    end
      
    rgb_data_uint=imread(rgb_name);
    rgb_data=im2double(rgb_data_uint);

    ConvertHStoHD5_31channel_31dim(rgb_data,hs_label,size_input,size_label,label_dimension,data_dimension,stride,chunksz,amount_hd5_image,filename)
   
 end        



