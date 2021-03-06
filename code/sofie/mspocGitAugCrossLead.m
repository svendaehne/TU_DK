 function [gamma,kappa_y,kappa_t,kappa_y0,kappa_t0, CrossEcut, corr_TEkf] = mspocGitAugCrossLead(Xe,Cxxe, Y_tr,G,Ne,Kf,Gamma,mspoc_params)


corr_TEkf = zeros(length(Gamma),length(mspoc_params.kappaY),length(mspoc_params.kappaT),Kf);
val_idx=0;
for kf=1:Kf
    fprintf('Starting fold %d; \n',kf);  

    val_idx = val_idx(end)+1:floor((Ne-val_idx(end))/(Kf+1-kf))+val_idx(end);
    tr=1:Ne;
    tr(val_idx)=[];
    opt.Cyy = [];
    opt.pca_Y_var_expl=0.95;
    y_tr = Y_tr(:,tr);
    [~, My_tr,Cyy_tr] = prepare_Y_signal(Y_tr(:,tr), opt);
    mspoc_params.Cxxe = Cxxe(:,:,tr);
    mspoc_params.Cyy = Cyy_tr;
    mspoc_params.My = My_tr;
    mspoc_params.verbose = 0;
    for kapt = 1:length(mspoc_params.kappaT)
        mspoc_params.kappa_tau = mspoc_params.kappaT(kapt);
    for kapy = 1:length(mspoc_params.kappaY)
        mspoc_params.kappa_y = mspoc_params.kappaY(kapy);
    for Gam= 1:length(Gamma);%0.999999;%0.001;
        %fprintf('Starting fold %d; gamma %d.\n',kf,Gam);  
        gamma=Gamma(Gam);
         rng(5)
        [wx, wy, wt, Ax_est, Ay_est, out] = mspocGitAugLead([], y_tr,G,gamma,mspoc_params);
       % mspoc_params
        %out.corr_values
        [corr_TEkf(Gam,kapy,kapt,kf), corr_TRkf(Gam,kapy,kapt,kf)] = calcCorrLead(Xe,Cxxe,Y_tr,wx,wy,wt,val_idx,[]);
    
        %calcCorrLead(Cxxe,Y_tr,wx,wy,wt,idx,Nt)
    end
    end
    end
end

scorr_TEkf=(sum(abs(corr_TEkf),4)/Kf);
[CrossEcut, im]=max(scorr_TEkf(:));
[igam,ikapY,ikapT] =ind2sub(size(scorr_TEkf),im);
%G_idxcut=im;
gamma=Gamma(igam);
kappa_y = mspoc_params.kappaY(ikapY);
kappa_t = mspoc_params.kappaT(ikapT);
fprintf('gamma found to be %d, kappa_y to be %d, kappa_tau to be %d\n',gamma,kappa_y,kappa_t)

scorr_TEkfg0=squeeze(scorr_TEkf(1,:,:));
[~, i0]=max(scorr_TEkfg0(:));
[ikapY0,ikapT0] =ind2sub(size(scorr_TEkfg0),i0);

kappa_y0 = mspoc_params.kappaY(ikapY0);
kappa_t0 = mspoc_params.kappaT(ikapT0);
fprintf('For gamma=0, kappa_y0 to be %d, kappa_tau0 to be %d\n',kappa_y0,kappa_t0)





function [Y_w, My,Cyy] = prepare_Y_signal(Y, opt)

% compute covariance matrix
[Ny, Ne] = size(Y);
if isempty(opt.Cyy)
    if Ne > Ny
        % if there are more samples than dimensions, compute the spatial
        % covariance matrix
        Cyy = Y*Y' / Ne;
    else
        % if there are more dimensions than samples, compute the temporal
        % covariance matrix
        Cyy = Y'*Y;
    end
else
    Cyy = opt.Cyy;
end

[V,D] = eig(Cyy);
[ev_sorted, sort_idx] = sort(diag(D), 'descend');
V = V(:,sort_idx);
My = V * diag(ev_sorted.^-0.5); % whitening filters are in the columns
% PCA dim-reduction
var_expl = cumsum(ev_sorted)./sum(ev_sorted);
min_var_expl = opt.pca_Y_var_expl; 
n = find(var_expl >= min_var_expl, 1, 'first');
My = My(:,1:n);


% whitening and possible dim-reduction
if Ne > Ny
    Y_w = My' * Y;
else
    Y_w = diag(std(V(:,1:n))) \ V(:,1:n)';
    My = sqrt(Ne) * Y * (V(:,1:n) * diag(1./ev_sorted(1:n)'));
end