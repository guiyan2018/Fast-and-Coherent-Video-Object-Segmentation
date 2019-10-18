clear
clc

f_F = Mean_F();
eval.F.mean   = mean(f_F,1);
eval.F.std    = std(f_F,1);
eval.F.recall = sum(f_F>0.5,1)/size(f_F,1);

tmp = get_mean_values(f_F,4);
eval.F.decay = tmp(1)-tmp(end);