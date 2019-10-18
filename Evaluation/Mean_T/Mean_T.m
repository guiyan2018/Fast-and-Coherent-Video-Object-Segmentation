function T = Mean_T()

% the second way for a video sequence
Path = 'G:\ѧ������\��������Segmentation\TCSVT\Revision\DAVIS\DAVIS\1\train';
VS = dir(Path);
imageNumber = 0;
for i = 1:size(VS,1)
    if not(strcmp(VS(i).name,'.')|strcmp(VS(i).name,'..')|strcmp(VS(i).name,'Thumbs.db'))
        imageNumber = imageNumber + 1; 
    end
end

Path1 = 'D:\2-Image_Video Object Segmentation\Video Object Segmentation\1-Video Database\DAVIS\Annotations\480p\train';
sum_T = 0;

for i = 1:imageNumber 
    %load the segmentation for each frame
    str = int2str(i);
    str = strcat('\', str, '.tif');
    str = strcat(Path,str);      
    mask = imread(str);
         
    %load ground truth for each frame
    str = int2str(i-1);
    if( (i-1)<10 )
        str = strcat('\', '0000', str, '.png');
    elseif ( (i-1)<100 )
        str = strcat('\', '000', str, '.png');
    else
        str = strcat('\', '00', str, '.png');
    end
    str = strcat(Path1,str);      
    gt = imread(str);
    
    %computing J index for each frame
    f_T = t_stability_single( mask, gt );
    sum_T = sum_T+f_T;
end

%Mean_T
T = sum_T/imageNumber;
