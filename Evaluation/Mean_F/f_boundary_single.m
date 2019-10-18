function F = f_boundary_single(foreground_mask, gt_mask, bound_th)


% Default threshold
if ~exist('bound_th','var')
    bound_th = 0.008;
end

% Get the amount of distance in pixels
if bound_th>=1
    bound_pix = bound_th;
else
    bound_pix = ceil(bound_th*sqrt(size(foreground_mask,1)^2+size(foreground_mask,2)^2));
end

% Get the pixel boundaries of both masks
fg_boundary = seg2bmap(foreground_mask);
gt_boundary = seg2bmap(gt_mask);

[X, Y] = meshgrid(-bound_pix:bound_pix,-bound_pix:bound_pix);
disk   =  (X .^ 2 + Y .^ 2) <= bound_pix .^ 2;

% Dilate them
fg_dil = imdilate(fg_boundary,strel(disk));
gt_dil = imdilate(gt_boundary,strel(disk));

% Get the intersection
gt_match = gt_boundary.*fg_dil;
fg_match = fg_boundary.*gt_dil;

% Area of the intersection
n_fg     = sum(fg_boundary(:));
n_gt     = sum(gt_boundary(:));

% Compute precision and recall
if (n_fg==0) && (n_gt>0)
    precision = 1;
    recall    = 0;
elseif (n_fg>0) && (n_gt==0)
    precision = 0;
    recall    = 1;
elseif (n_fg==0) && (n_gt==0)
    precision = 1;
    recall    = 1;
else
    precision = sum(fg_match(:))/n_fg;
    recall    = sum(gt_match(:))/n_gt;
end

% Compute F measure
if (precision+recall==0)
    F = 0;
else
    F = 2*precision*recall/(precision+recall);
end


