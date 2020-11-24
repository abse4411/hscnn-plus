function filename_change=Get_filename(filename,prefix,folder_label)

filename_change=filename;
filename_change=[prefix filename_change];
filename_change=filename_change(1:end-4);
filename_change=strcat(filename_change,'.h5');


