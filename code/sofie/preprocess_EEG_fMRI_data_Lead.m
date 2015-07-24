function [X,Y,Cxxe, info] = preprocess_EEG_fMRI_data_Lead(X,Y, eeg_sf, ssd_freqs, fmri_sf, varargin)


display('Starting EEG/fMRI preprocessing')

%% input params
opt = propertylist2struct(varargin{:});
opt = set_defaults(opt, ...
    'ssd_freqs', 10, ...
    'n_ssd_components', 20, ...
    'upsample_factor', 1, ...
    'fmri_hp_ival', 2*60, ...
    'fmri_lp_cutoff', 0.2, ...
    'fmri_mask', [], ...
    'data_info', [], ...
    'run_ssd', 1, ...
    'verbose', 1);

%% EEG preprocessing parameters
ssd_components = 1:opt.n_ssd_components;
if isscalar(ssd_freqs)
    fc = ssd_freqs;
    ssd_freqs = [
        2.^(log2(fc)+[-0.25, 0.25]);
        2.^(log2(fc)+[-0.8, 0.8]);
        2.^(log2(fc)+[-0.4, 0.4]);
        ];
end

%% fMRI parameters
fmri_hp_ival = opt.fmri_hp_ival;
fmri_hp_cutoff = 1/fmri_hp_ival;
fmri_lp_cutoff = opt.fmri_lp_cutoff;
mask = opt.fmri_mask;
if isempty(mask)
    warning('No fMRI mask provided, so all voxels will be used! Consider using the SPM gray matter mask, for example!');
    mask = ones(size(Y(:,:,:,1)));
end

%% other parameters
upsample_factor = opt.upsample_factor;
verbose = opt.verbose;

info = opt.data_info;
info.preprocessing_opt = opt;

%% sanity check

% get dimensions of EEG data
[Te, Nx, Ne] = size(X);
if not(Ne == size(Y,4))
    error('Number of EEG epochs does not match number of fMRI volumens!');
end

%% extract mask voxel timecourses from fMRI and subtract mean 

Y = reshape(Y, [numel(Y(:,:,:,1)), size(Y,4)]);
Y = Y(mask(:),:);
Y = bsxfun(@minus, Y, mean(Y,2));

%% filter the fMRI

display('  -filtering the fMRI data')

% high-pass filter
[b,a] = butter(5, fmri_hp_cutoff/fmri_sf*2, 'high');
Y = filtfilt(b,a,Y')';

% low-pass filter
[b,a] = butter(5, fmri_lp_cutoff/fmri_sf*2);
Y = filtfilt(b,a,Y')';

%% upsample EEG and fMRI

Y = resample(Y', upsample_factor, 1)';
info.fmri.fs = fmri_sf*upsample_factor;

X = reshape(permute(X, [1,3,2]), [Te*Ne, Nx]);
Ne = upsample_factor * Ne;
Te = Te / upsample_factor;
X = permute(reshape(X, [Te, Ne, Nx]), [1,3,2]);

% create an index vector, so that we later know which epochs are still
% there and which ones have been removed. 
idx_vec = 1:Ne;

%%
display('  -removing broadband EEG outlier epochs')
rm_idx = detect_high_variance_epochs(squeeze(log(var(X))), [],verbose,2);

%%
X(:,:,rm_idx) = [];
Y(:,rm_idx) = [];
idx_vec(rm_idx) = [];

%% SSD denoising/dim-reduction
display('  -computing SSD denoising')

X = permute(X, [1,3,2]);
X = reshape(X, [size(X,1)*size(X,2), size(X,3)]);
[~, A_ssd, lambda_ssd, ~, X_ssd] = ssd(X, ssd_freqs, eeg_sf,[],[]);

info.preprocessing_opt.A_ssd = A_ssd;
info.preprocessing_opt.lambda_ssd = lambda_ssd;
% SSD-denoising by projecting only a subset of components back to channel
% space

X = X_ssd(:,ssd_components) * A_ssd(:,ssd_components)';

%% compute covariance matrix time-series (and median filter it)
display('  -computing EEG cov-timeseries')
X_epo = permute(reshape(X, [Te, size(X,1)/Te, size(X,2)]), [1,3,2]);
Cxxe = epochwise_covariance(X_epo);
[X_epo, idx_flt, Cxxe] = median_filter_cov_timeseries(Cxxe,3, verbose);
%% one more step of EEG outlier detection
display('  -removing EEG outlier epochs')
rm_idx = detect_max_eigenvalue_epochs(Cxxe,[],verbose,2);
rm_idx1=rm_idx;
%%
Cxxe(:,:,rm_idx) = [];
Y(:,rm_idx) = [];
X_epo(:,:,rm_idx)=[];
idx_vec(rm_idx) = [];

%% detect fMRI artifact epochs

display('  -removing fMRI outlier epochs')
rm_idx = detect_high_variance_epochs(Y,[],verbose,2);
rm_idx2=rm_idx;
%%
Cxxe(:,:,rm_idx) = [];
Y(:,rm_idx) = [];
X_epo(:,:,rm_idx)=[];
idx_vec(rm_idx) = [];
info.Xe=X_epo;
%% detect bad fMRI voxels

display('  -detecting fMRI outlier voxels')

% compute bad-voxel treshold based on the distribution of variance
V = var(Y,[],2);
p = percentiles(V, [50,95]);
threshold = p(1) + 10*(p(2)-p(1));
bad_chan = V > threshold;



%% remove time-courses of bad voxels
mask2 = bad_chan == 0;
Y = Y(mask2,:);
mask(mask) = not(bad_chan);
info.fmri.mask = mask;
info.preprocessing_opt.bad_voxel_idx = bad_chan;
%% repeat for non-ssd-reduced EEG
if opt.noRed==1
X_noRed = X_ssd * A_ssd';
X_epo_noRed = permute(reshape(X_noRed, [Te, size(X_noRed,1)/Te, size(X_noRed,2)]), [1,3,2]);
Cxxe_noRed = epochwise_covariance(X_epo_noRed);
Cxxe_noRed = median_filter_cov_timeseries(Cxxe_noRed,3, verbose);
Cxxe_noRed(:,:,rm_idx1) = [];
Cxxe_noRed(:,:,rm_idx2) = [];
info.Cxxe_noRed=Cxxe_noRed;
end
%% done

info.preprocessing_opt.idx_vector = idx_vec;
display('done!')