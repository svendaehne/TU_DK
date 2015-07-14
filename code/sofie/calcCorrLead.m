function [corr_TE] = calcCorrLead(Cxxe,Y,wx,wy,wt,idx)
% Inputs:
%   idx - validation set indices
Nt = length(wt);
Y=bsxfun(@minus, Y, mean(Y,2));
if idx(1)==1
    px_idx=idx(Nt+1:end);
else
    px_idx=idx;
end
Ne = size(Y,2);
sy_est = wy' * Y;
% - X estimated power(variance) of source
Nx = size(Cxxe,1);
Cxxe_vec = reshape(Cxxe, [Nx*Nx, Ne]);
wx_vec = reshape(wx*wx',[Nx*Nx,1]);
px_est = wx_vec' * Cxxe_vec;

% - filter X source with temporal filter (should mimic hrf)
wt_tmp = wt/sum(wt);
px_flt_est = filter(wt_tmp, 1, px_est);

% corr temp, est filt sx and sy_est, test
corrtmp = corrcoef(sy_est(px_idx), px_flt_est(px_idx));
corr_TE = (corrtmp(1,2));
