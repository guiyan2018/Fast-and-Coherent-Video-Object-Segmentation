function f_J = Mean_J()

% the second way for a video sequence
Path = 'D:\2-Image_Video Object Segmentation\Video Object Segmentation\1-Video Database\DAVIS\Annotations\480p\tennis';

VS = dir(Path);
imageNumber = 0;
for i = 1:size(VS,1)
    if not(strcmp(VS(i).name,'.')|strcmp(VS(i).name,'..')|strcmp(VS(i).name,'Thumbs.db'))
        imageNumber = imageNumber + 1; 
    end
end

% imageNumber = 50;

f_J = zeros(imageNumber, 1);
% Path1 = 'G:\学术论文\论文三：Segmentation\Supplemental materials\Reasults\DAVIS 2016\car-shadow';
Path1 = 'G:\学术论文\论文三：Segmentation\TCSVT\Revision\DAVIS\DAVIS\2\tennis';
for i = 1:imageNumber 
    %load ground truth for each frame
    str = int2str(i-1);
    if( (i-1)<10 )
        str = strcat('\', '0000', str, '.png');
    elseif( (i-1)<100 )
        str = strcat('\', '000', str, '.png');
    else
        str = strcat('\', '00', str, '.png');
    end
    str = strcat(Path,str);      
    gt = imread(str);
    
    if( size(gt,3) >= 3)  
        gt = gt(:,:,1);
    end
    
    %load the segmentation for each frame
%     str = int2str(i-1);
%     if( (i-1)<10 )
%         str = strcat('\', '0000', str, '.png');
%     elseif( (i-1)<100 )
%         str = strcat('\', '000', str, '.png');
%     else
%         str = strcat('\', '00', str, '.png');
%     end
    str = int2str(i);
    str = strcat('\', str, '.tif');
    str = strcat(Path1,str);      
    mask = imread(str);
    
    if( size(mask,3) >= 3)  
        mask = mask(:,:,1);
    end
         
    %computing J index for each frame
    f_J(i,1) = jaccard_single( mask, gt );
end


