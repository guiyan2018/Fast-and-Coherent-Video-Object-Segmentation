clear
clc

f_J = Mean_J();
% V = mean(f_J,1);

eval.J.mean   = mean(f_J,1);
eval.J.std    = std(f_J,1);
eval.J.recall = sum(f_J>0.5,1)/size(f_J,1);

tmp = get_mean_values(f_J,4);
eval.J.decay = tmp(1)-tmp(end);