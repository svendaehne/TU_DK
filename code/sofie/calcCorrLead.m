function [corr_TE, corr_TR] = calcCorrLead(Xe,Cxxe,Y,wx,wy,wt,idx,tr_idx)
% Inputs:
%   idx - validation set indices
Nt = length(wt);
Ne = size(Y,2);
mask=Nt:Ne;
val_idx=intersect(idx,mask);
if isempty(tr_idx)
tr_idx = Nt:Ne;
tr_idx=setdiff(tr_idx,val_idx);
end
Y=bsxfun(@minus, Y, mean(Y,2));

sy_est = wy' * Y;
%sy_est = (sy_est - mean(sy_est(tr_idx)))/std(sy_est(tr_idx));

% - X estimated power(variance) of source
Nx = size(Cxxe,1);
Cxxe_vec = reshape(Cxxe, [Nx*Nx, Ne]);
wx_vec = reshape(wx*wx',[Nx*Nx,1]);
px_est = wx_vec' * Cxxe_vec;
px_est = px_est-mean(px_est);

% - filter X source with temporal filter (should mimic hrf)
wt_tmp = wt/sum(wt);
px_flt_est = filter(wt_tmp, 1, px_est);


keyboard
% keyboard
% %%
px_est2 = zeros(1,Ne);
for e=1:Ne
    px_est2(e) = var(Xe(:,:,e) * wx);
end
% - filter X source with temporal filter (should mimic hrf)
%wt_tmp = wt/sum(wt);
px_flt_est2 = filter(wt_tmp, 1, px_est2);
%%
%keyboard
% corr temp, est filt sx and sy_est, test
corrtmp = corrcoef(sy_est(val_idx), px_flt_est2(val_idx));
corr_TE = (corrtmp(1,2));

% corr temp, est filt sx and sy_est, train
corrtmp = corrcoef(sy_est(tr_idx), px_flt_est2(tr_idx));
corr_TR = (corrtmp(1,2));


