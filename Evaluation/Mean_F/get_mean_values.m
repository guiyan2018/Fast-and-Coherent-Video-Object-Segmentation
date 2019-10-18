function mvals = get_mean_values(values,N_bins)
    % Get four mean values to see how the quality evolves with time
    ids = round(linspace(1, length(values),N_bins+1));
    mvals = zeros(1,length(ids)-1);
    for jj=1:(length(ids)-1)
       mvals(jj) = mean(values(ids(jj):ids(jj+1)));
    end
end