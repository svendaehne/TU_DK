function fig_h = viz_mspoc_components(Wx, Wy, Wtau, Ax, Ay, mspoc_out, Cxxe, Y, info)


%% params
[~, Nx, Ne] = size(Cxxe);

n_components = size(Wx,2);
mnt = getElectrodePositions(info.eeg.clab);
sc_opt = my_scalpMap_opt;

fmri_fs = info.fmri.fs;
mask = info.fmri.mask;
tau_vector = mspoc_out.tau_vector;

%% compute component time-courses
Sy = Wy'*Y;
Cxxe_vec = reshape(Cxxe, [Nx*Nx, Ne]);

t_vec = tau_vector/fmri_fs;

max_lag = ceil(30 * fmri_fs);

px_flt = zeros(n_components, Ne);
px = zeros(n_components, Ne);
xcorrs = zeros(n_components, 2*max_lag+1);
for k=1:n_components
    wx_vec = reshape(Wx(:,k)*Wx(:,k)', [size(Wx,1)^2, 1]);
    px(k,:) = wx_vec'*Cxxe_vec;
    px(k,:) = log(px(k,:));
    [xcorrs(k,:), lags] = xcorr(Sy(k,:)', px(k,:)', max_lag, 'coeff');

    px_flt(k,:) = filter(Wtau(:,k), 1, px(k,:)')';
end


%% plot patterns and time-courses
fig_h = zeros(n_components,3);
for k=1:n_components
    
    fig_h(k,1) = figure;
    
    rows = 1;
    cols = 3;
    
    subplot(rows,cols,1)
    pat = Ax(:,k);
    [~,mm_idx] = max(abs(pat));
    pat = pat * sign(pat(mm_idx));
    scalpPlot(mnt, pat, sc_opt);
    title({'spatial pattern',sprintf('of EEG component #%d',k)})
%     colorbar
    
    subplot(rows,cols,2)
    hold on
    plot(lags, zeros(1,length(lags)), 'color', 0.8*ones(1,3))
    plot(lags/fmri_fs, xcorrs(k,:), 'k', 'linewidth',2)
    plot([0,0],[-1,1], 'color', 0.8*ones(1,3))
    title({'cross-correlation between', ...
        sprintf('power dynamics of EEG component #%d',k), ...
        sprintf('and time-course of fMRI component #%d',k)})
    ylabel('r')
    xlabel('time-lag [s]')
    xlim([lags(1), lags(end)])
    ylim(max(abs(xcorrs(k,:)))*1.1*[-1,1])
    box on
    
    subplot(rows,cols, 3)
    hold on
    plot(t_vec, zeros(1,length(t_vec)), 'color', 0.8*ones(1,3))
%     plot(t_vec,zscore([Wtau(:,k), mspoc_out.Atau(:,k)]))
    plot(tau_vector, Wtau(:,k),'k', 'linewidth',2)
    title('estimated FIR filter')
    xlabel('time-lag [s]')
    xlim([t_vec(1), t_vec(end)])
    ylabel('[a.u.]')
    box on
    
    fig_h(k,2) = figure;
    %     subplot(rows,cols,cols+(1:cols))
    hold on
    t = (1:Ne)/fmri_fs;
    plot(t, zscore(px(k,:)), 'b')
    plot(t, zscore([px_flt(k,:)', Sy(k,:)']), 'linewidth',2)
    xlim([t(1), t(end)])
    xlabel('time [s]')
    ylabel('[s.d.]')
    title('time-courses of component activation')
    legend({'EEG component power','EEG component power, convolved', 'fMRI component'})
    box on
    
    M = zeros(size(mask));
    M(mask) = Ay(:,k);
    fig_h(k,3) = figure;
%     M = permute(M, [2,1,3]);
    plot_brain2d(M, 4,6,3, max(abs(M(:)))*[-1,1]);
    colormap(cmap_posneg(101));
end
