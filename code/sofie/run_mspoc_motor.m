
%% params
cd C:\Users\sofha\SPoC\TU_DK\code\sofie
addpath C:\Users\sofha\SPoC\matlab_SPoC-master\OriginalCode\plot_brains
addpath C:\Users\sofha\SPoC\TU_DK\code\sven
addpath C:\Users\sofha\SPoC\matlab_SPoC\utils
addpath C:\Users\sofha\SPoC\matlab_SPoC\SSD

data_folder = 'C:\Users\sofha\SPoC\matlab_SPoC-master\Data';
run_name = 'motor_execution';freq=20;
%run_name = 'eyes_open_closed';freq=10;
% freq = [
%     2.^(log2(freq)+[-0.25, 0.25]);
%     2.^(log2(freq)+[-0.8, 0.8]);
%     2.^(log2(freq)+[-0.4, 0.4]);
%     ]


%% load data
switch run_name
    case 'motor_execution'
        tmp = load(fullfile(data_folder, 'motor_data.mat'));
    case 'eyes_open_closed'
        tmp = load(fullfile(data_folder, 'eyesOpenClosed_data.mat'));
end

X = tmp.X;
Y = tmp.Y;
data_info = tmp.dat_info;
clear tmp

%%
% HERE WE HAVE TO CHANGE THE ORDER OF DIMS SO THAT IT CORRESPONDS TO
% DANIELS STUFF
Y = permute(Y,[2,1,3,4]);
data_info.fmri.dim = data_info.fmri.dim([2,1,3]);
data_info.fmri.mask = permute(data_info.fmri.mask, [2,1,3]);

eeg_sf = data_info.eeg.fs;
fmri_sf = data_info.fmri.fs;
mask = data_info.fmri.mask;

%% load Daniel's brain mask and leadfield


% modify the grey matter mask such that voxels
% outside the brain are not included anymore
tmp = load(fullfile(data_folder,'brainmask.mat'));
brain_mask = tmp.brainmask;
M = zeros(size(mask));
M(mask) = brain_mask;

% figure;
% plot_brain2d(M, 4,6,3, max(abs(M(:)))*[-1,1]);

M = M > 0;

tmp = load(fullfile(data_folder,'leadfieldSofieANDSven.mat'));
L = tmp.L(:,brain_mask,:);
%% preprocess data

[Xr, Yr, Cxxe, info] = preprocess_EEG_fMRI_data_Lead(X,Y, eeg_sf, freq, fmri_sf,...
    'fmri_mask', M,...
    'n_ssd_components', 20, ...
    'upsample_factor', 2, ...
    'fmri_PCA_var_expl', 0.99, ...
    'verbose', 0, ...
    'noRed', 1, ...
    'data_info', data_info);
upsample_factor = info.preprocessing_opt.upsample_factor;
Cxx = squeeze(mean(Cxxe,3));
Cxx_All = squeeze(mean(info.Cxxe_noRed,3));

% remove outlier voxels from leadfield
L(:,info.preprocessing_opt.bad_voxel_idx,:) = [];

%% sanity check

% make sure the number of voxels in the fMRI corresponds to the number
% dipole locations in the leadfield

if not(size(L,2) == size(Yr,1))
    error('Nr of voxels does not match number of dipole locs in the leadfield!')
end

%% reduce size of L
iCxx = inv(Cxx_All);
Lnew=NaN(size(L,1),size(L,2));
for ny=1:size(L,2);
Lcov = squeeze(L(:,ny,:))'*iCxx*squeeze(L(:,ny,:));
[eigV,SCORE] = pcacov(Lcov);
Lnew(:,ny)=eigV(1)*L(:,ny,1)+eigV(2)*L(:,ny,2)+eigV(3)*L(:,ny,3);
end
%% mspoc opts
tau_vector = 0:info.fmri.fs:20;
[~, Nx, Ne] = size(Cxxe);
Cxxe_vec = reshape(Cxxe, [Nx*Nx, Ne]);


mspoc_opt = struct(...
    'Cxxe', Cxxe,...
    'tau_vector', tau_vector, ... % maximum timeshift of X relative to Y, given in # epochs
    'use_log', 1, ...
    'n_random_initializations', 10, ...
    'max_optimization_iterations', 20, ...
    'pca_Y_var_expl', 0.95, ...
    'verbose', 0 ...
    );

%% optimize regularizers

best_kappa_tau = [];
best_kappa_y = [];

[best_kappa_tau, best_kappa_y, out] = ...
    optimize_mspoc_regularizersLead([], Yr, mspoc_opt, ...
        'n_xvalidation_folds', 4 , ...
        'kappa_tau_list', 10.^(-3:3), ...
        'kappa_y_list', 10.^(-3:3), ...0:0.1:1, ...
        'Cxxe', Cxxe);
R_tr = mean(out.corr_tr,3);
R_te = mean(out.corr_te,3);

%%

figure
rows = 1;
cols = 3;
subplot(rows,cols,1)
imagesc(R_tr);%, [0,1])
colorbar
title('correlation on training data')

subplot(rows,cols,2)
imagesc(R_te);%, [0,1])
colorbar
title('correlation on test data')

subplot(rows,cols,3)
imagesc(R_te+R_tr);%, [0,1])
colorbar
title('sum of correlations on training and test data')
%% Find optimal gamma
n_components = 1;

mspoc_opt.n_component_sets = n_components;
mspoc_opt.verbose = 2;
mspoc_opt.kappa_tau = 1e-3;%best_kappa_tau;%
mspoc_opt.kappa_y = 1;%best_kappa_y;%1

%mspoc_opt.kappa_tau = 10.^-1;
%KappaY=logspace(-2,2,5);

mspoc_opt.kappaY = logspace(-2,2,5);
mspoc_opt.kappaT = logspace(-3,2,6);
Gamma=linspace(0,1,10);
Kf=4;
No_Ne=100:100:800;

for ne=1:length(No_Ne)
    fprintf('runnning epoch %d out of 4\n',ne)
mspoc_opt.Cxxe = Cxxe(:,:,1:No_Ne(ne));
epochs= 1:No_Ne(ne);
clear mspoc_opt.Cxx;
Yr_ne = Yr(:,epochs);
Ne_new = size(Yr_ne,2);
tic_cross=tic;
[gamma(ne),kappa_y(ne),kappa_t(ne),kappa_y0(ne),kappa_t0(ne), CrossEcut, corr_TEkf]=mspocGitAugCrossLead(info.Xe(:,:,epochs),Cxxe(:,:,epochs),Yr_ne,Lnew,Ne_new,Kf,Gamma,mspoc_opt);
gamma(ne)
toc(tic_cross)
% mspoc_opt.kappa_y=kappa_y(ne);
 [Wx, Wy, Wt, Ax, Ay, mspoc_out] = mspocGitAugLead([], Yr_ne,Lnew,gamma(ne),mspoc_opt);
 [corr_TE(ne), corr_TR(ne)] = calcCorrLead(info.Xe,Cxxe,Yr,Wx,Wy,Wt,800:1063,epochs);
% 
 mspoc_opt.kappa_y=kappa_y0(ne);
 Xe=reshape(X,400,25,540);
 [Wx, Wy, Wt, Ax, Ay, mspoc_out] = mspocGitAugLead([], Yr_ne,[],0,mspoc_opt);
 [corr_TE0(ne), corr_TR0(ne)] = calcCorrLead(info.Xe,Cxxe,Yr,Wx,Wy,Wt,800:1063,epochs);

end

%% run mspoc
n_components = 1;

mspoc_opt.n_component_sets = n_components;
mspoc_opt.verbose = 2;
%mspoc_opt.kappa_tau = best_kappa_tau;
%mspoc_opt.kappa_y = best_kappa_y;

mspoc_opt.kappa_tau = 10.^-1;
mspoc_opt.kappa_y = 10.^1;

mspoc_opt.Cxxe = Cxxe;%(:,:,1:200)
%[Wx, Wy, Wtau, Ax, Ay, mspoc_out] = mspocGit([], Yr, mspoc_opt);
[Wx, Wy, Wt, Ax, Ay, mspoc_out] = mspocGitAugLead([], Yr,Lnew,0.2,mspoc_opt);


%% plot results
mspoc_out.tau_vector = mspoc_opt.tau_vector;
fig_h = viz_mspoc_components(Wx, Wy, Wt, Ax, Ay, mspoc_out, Cxxe, Yr, info);
%%
load('C:\Users\sofha\SPoC\matlab_SPoC-master\EEGpos2D')
for nc=1:n_components;
[ZI,f] = spm_eeg_plotScalpData(Ax(:,nc),xy,EEGlab);
caxis([-max(abs(Ax(:,nc))),max(abs(Ax(:,nc)))])
end
%%
Axy=squeeze(L(:,:,1))*Ay;
Axy=Axy/max(abs(Axy));
[ZI,f] = spm_eeg_plotScalpData(Axy,xy,EEGlab);
caxis([-max(abs(Axy(:,nc))),max(abs(Axy(:,nc)))])
%% save results

results = [];
results.Wx = Wx;
results.Wy = Wy;
results.Wtau = Wtau;
results.Ax = Ax;
results.Ay = Ay;
results.info = info;

Wy_tmp = zeros([prod(info.fmri.dim), n_components]);
Wy_tmp(info.fmri.mask(:),:) = Wy;
results.Sy = Wy_tmp' * reshape(Y, [prod(info.fmri.dim), size(Y,4)]);

if not(exist(fullfile(result_path, run_name), 'dir'))
    mkdir(fullfile(result_path, run_name));
end
save(fullfile(result_path, run_name, sprintf('fc_%d__mspoc_results.mat',freq)), 'results');

% save fMRI pattern as .nii file
for k=1:n_components
    sparsify = 1;
    if sparsify
        Ay_z = zscore(Ay(:,k));
        Ay_thresh = Ay(:,k);
        Ay_thresh(abs(Ay_z) <= 3.5)=0;
    else
        Ay_thresh = Ay(:,k);
    end
    
    Ay_thresh = Ay_thresh / max(abs(Ay_thresh));
    
    M = zeros(info.fmri.dim);
    M(info.fmri.mask(:)) = Ay_thresh;
    
    fig_name = sprintf('fc_%d_component_%d__mspoc_fmri_pattern',freq,k);
    if sparsify
        fig_name = [fig_name '_sparse'];
    end
    
    % save volume
    tmp_h = info.fmri.hdr(1);
    tmp_h = rmfield(tmp_h, 'pinfo');
    tmp_h.fname = [fullfile(result_path, run_name, fig_name) '.nii'];
    spm_write_vol(tmp_h,M);
end
%%
n_components = 1;

mspoc_opt.n_component_sets = n_components;
mspoc_opt.verbose = 2;
mspoc_opt.kappa_tau = best_kappa_tau;
mspoc_opt.kappa_y = best_kappa_y;

mspoc_opt.kappa_tau = 10.^-2;
KappaY=1;%logspace(-2,2,5);

mspoc_opt.kappaY = logspace(-2,2,5);
Gamma=linspace(0,1,10);
Kf=4;
corr_TEKF5 = zeros(Kf,1);
corr_TRKF5 = zeros(Kf,1);
corr_TEKF0 = zeros(Kf,1);
corr_TRKF0 = zeros(Kf,1);

val_idx=0;
for kf=1:Kf  
    val_idx = val_idx(end)+1:floor((Ne-val_idx(end))/(Kf+1-kf))+val_idx(end);
    tr=1:Ne;
    tr(val_idx)=[];
    opt.Cyy = [];
    opt.pca_Y_var_expl=0.95;
    y_tr = Yr(:,tr);
    Ne_new = length(tr);
    mspoc_params.Cxxe = Cxxe(:,:,tr);
   mspoc_opt.Cxxe = Cxxe(:,:,tr);

    mspoc_params.verbose = 0;
clear mspoc_opt.Cxx;
tic_cross=tic;
[gamma(kf),kappa_y(kf),kappa_y0(kf), CrossEcut, corr_TEkf]=mspocGitAugCrossLead(Cxxe(:,:,tr),y_tr,L,Ne_new,Kf,Gamma,mspoc_opt);
gamma(kf)
toc(tic_cross)
mspoc_opt.kappa_y=kappa_y(ne);
[Wx, Wy, Wt, Ax, Ay, mspoc_out] = mspocGitAugLead([], y_tr,L,gamma(kf),mspoc_opt);
[corr_TEKF5(kf), corr_TRKF5(kf)] = calcCorrLead(Cxxe,Yr,Wx,Wy,Wt,val_idx,[]);

mspoc_opt.kappa_y=kappa_y0(ne);
[Wx, Wy, Wt, Ax, Ay, mspoc_out] = mspocGitAugLead([], y_tr,[],0,mspoc_opt);
[corr_TEKF0(kf), corr_TRKF0(kf)] = calcCorrLead(Cxxe,Yr,Wx,Wy,Wt,val_idx,[]);

end
