
%% params

data_folder = '/home/bbci/data/daehne/EEG_fMRI_MPI_Leipzig/EEG-fMRI_Benchmark/preprocessed_EEG_fMRI_matfiles/';

% run_name = 'motor_execution';
run_name = 'eyes_open_closed';


% frequency of interest
freq = 20;
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

[Xr, Yr, Cxxe, info] = preprocess_EEG_fMRI_data(X,Y, eeg_sf, freq, fmri_sf,...
    'fmri_mask', M,...
    'n_ssd_components', 20, ...
    'upsample_factor', 2, ...
    'verbose', 0, ...
    'data_info', data_info);
upsample_factor = info.preprocessing_opt.upsample_factor;

% remove outlier voxels from leadfield
L(:,info.preprocessing_opt.bad_voxel_idx,:) = [];

%% sanity check

% make sure the number of voxels in the fMRI corresponds to the number
% dipole locations in the leadfield

if not(size(L,2) == size(Yr,1))
    error('Nr of voxels does not match number of dipole locs in the leadfield!')
end





%% mspoc opts
tau_vector = 0:20;

mspoc_opt = struct(...
    'tau_vector', tau_vector, ... % maximum timeshift of X relative to Y, given in # epochs
    'use_log', 1, ...
    'n_random_initializations', 10, ...
    'max_optimization_iterations', 20, ...
    'pca_Y_var_expl', 0.99, ...
    'verbose', 0 ...
    );

%% optimize regularizers
kappa_tau_list = 10.^1; ...(-2:1:2);
kappa_y_list = 10.^(-2:1:2);
    
if length(kappa_tau_list) == 1 && length(kappa_y_list) ==1
    best_kappa_tau = kappa_tau_list(1);
    best_kappa_y = kappa_y_list(1);
else
    [best_kappa_tau, best_kappa_y] = ...
        optimize_mspoc_regularizers([], Yr, mspoc_opt, ...
        'n_xvalidation_folds', 3 , ...
        'kappa_tau_list', kappa_tau_list, ...
        'kappa_y_list', kappa_y_list, ...
        'Cxxe', Cxxe);
end

%% run mspoc on training data
n_components = 1;
mspoc_opt.n_component_sets = n_components;
mspoc_opt.verbose = 1;
mspoc_opt.Cxxe = Cxxe;
mspoc_opt.kappa_tau = best_kappa_tau;
mspoc_opt.kappa_y = best_kappa_y;

[Wx, Wy, Wtau, Ax, Ay] = mspoc([], Yr, mspoc_opt);


%% plot results on test data

% mspoc_out.tau_vector = mspoc_opt.tau_vector;
% info.fmri.mask = M;
fig_h = viz_mspoc_components(Wx, Wy, Wtau, Ax, Ay, mspoc_opt, Cxxe, Yr, info);

%%

[max_values, sort_idx] = sort(Ay,'descend');

sc_opt = my_scalpMap_opt;
mnt = getElectrodePositions(info.eeg.clab);

% max_idx = randi(size(Ay,1));
indices = sort_idx(1:5);
figure
rows = length(indices);
cols = 3;
for n=1:length(indices)
    for k=1:3
        subplot(rows,cols,(n-1)*cols + k)
        scalpPlot(mnt, squeeze(L(:,indices(n),k)), sc_opt);
    end
end



% %% split off test data
% 
% idx_te = 721:size(Yr,2);
% Cxxe_te = Cxxe(:,:,idx_te);
% Y_te = Yr(:,idx_te);
% 
% %% test amount of training data necessary
% 
% n_repeats = 10;
% train_secs = [30,60:60:600];
% 
% n_repeats = 1;
% train_secs = [700];
% 
% N = length(train_secs);
% r = zeros(n_repeats, N);
% for n=1:N
%     
%     %% repeat loop
%     for m=1:n_repeats
%         
%         %% pick training data
%         n_training_sec = train_secs(n);
%         
%         max_tr_idx = 720;
%         start_idx = randperm(max_tr_idx-n_training_sec+1, 1);
%         stop_idx = start_idx + n_training_sec - 1;
%         
%         idx_tr = start_idx:stop_idx;
%         Cxxe_tr = Cxxe(:,:,idx_tr);
%         Y_tr = Yr(:,idx_tr);
%         
%         %% this should be paralllized if possible!
%         [r(m,n), out] = train_test_mSPoC(Cxxe_tr,Y_tr, Cxxe_te, Y_te);
%         Wx = out.Wx;
%         Wy = out.Wy;
%         Wtau = out.Wtau;
%         Ax = out.Ax;
%         Ay = out.Ay;
%         mspoc_opt = out.mspoc_opt;
%         fprintf(' resulting correlation on test data: %g\n',r(m,n))
%     end
% end
% 
% fprintf('\n\n\n DONE. \n')
