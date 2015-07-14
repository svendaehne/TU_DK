 function [gamma, CrossEcut, corr_TEkf] = mspocGitAugCrossLead(Cxxe, Y_tr,G, hrf, kappa_tau,kappa_y,Ne,Kf,Gamma,rep)


corr_TEkf = zeros(length(Gamma),Kf);
val_idx=0;
for kf=1:Kf
    
    val_idx = val_idx(end)+1:floor((Ne-val_idx(end))/(Kf+1-kf))+val_idx(end);
    tr=1:Ne;
    tr(val_idx)=[];
    y_tr = Y_tr(:,tr);
    Cxxe_tr = Cxxe(:,:,tr);
    for Gam= 1:length(Gamma);%0.999999;%0.001;
        fprintf('Starting fold %d; gamma %d.\n',kf,Gam);  
        gamma=Gamma(Gam);
         rng(rep)
        [wx, wy, wt, Ax_est, Ay_est, out] = mspocGitAugLead([], y_tr,G,gamma,...
            'tau_vector', hrf, 'kappa_tau', kappa_tau, 'kappa_y', kappa_y,'Cxxe',Cxxe_tr,'verbose',0);
        [corr_TEkf(Gam,kf)] = calcCorrLead(Cxxe,Y_tr,wx,wy,wt,val_idx);
        %calcCorrLead(Cxxe,Y_tr,wx,wy,wt,idx,Nt)
    end
end
keyboard
[CrossEcut, im]=max((sum(abs(corr_TEkf'))/Kf));
%G_idxcut=im;
gamma=Gamma(im);