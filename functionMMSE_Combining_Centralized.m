function [V_MMSE_Combining] = functionMMSE_Combining_Centralized(Hhat,C_tot,nbrOfRealizations,M,N,K,p)
%%=============================================================
%The file is used to design centralized MMSE combining vectors of the paper:
%
%Z. Wang, J. Zhang, E. Björnson, D. Niyato, and B. Ai, "Optimal Bilinear Equalizer for Cell-Free Massive MIMO Systems over Correlated Rician Channels," 
%in IEEE Transactions on Signal Processing, 2025, doi: 10.1109/TSP.2025.3547380.
%
%Download article: https://arxiv.org/abs/2407.18531 or https://ieeexplore.ieee.org/document/10920478
%
%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%paper as described above.
%%============================================================

eyeMN = eye(M*N);



%Diagonal matrix with transmit powers and its square root
Dp = diag(p);



V_MMSE_Combining = zeros(M*N,K,nbrOfRealizations);


%% Go through all channel realizations
for n = 1:nbrOfRealizations
  
    %Extract channel estimate realizations from all UEs to all APs
    Hhatallj = reshape(Hhat(:,n,:),[M*N K]);
    
    
    %Compute MMSE combining
    V_MMSE_Combining(:,:,n) = ((Hhatallj*Dp*Hhatallj') + C_tot + eyeMN)\(Hhatallj*Dp);
   

end

V_MMSE_Combining = permute(V_MMSE_Combining,[1 3 2]);