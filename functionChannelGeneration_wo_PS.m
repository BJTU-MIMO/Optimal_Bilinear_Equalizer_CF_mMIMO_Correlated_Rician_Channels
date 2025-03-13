function [H,H_LoS] = functionChannelGeneration_wo_PS(R_AP,HMean_Withoutphase,M,K,N,nbrOfRealizations)
%%=============================================================
%The file is used to generate the channel without random phase-shifts in LoS components of the paper:
%
%Z. Wang, J. Zhang, E. Björnson, D. Niyato, and B. Ai, "Optimal Bilinear Equalizer for Cell-Free Massive MIMO Systems over Correlated Rician Channels," 
%in IEEE Transactions on Signal Processing, 2025, doi: 10.1109/TSP.2025.3547380.
%
%Download article: https://arxiv.org/abs/2407.18531 or https://ieeexplore.ieee.org/document/10920478
%
%License: This code is licensed under the GPLv2 license. If you in any way
%use this code for research that results in publications, please cite our
%paper as described above.
%%=============================================================

%Prepare to store the results   
R = zeros(M*N,M*N,K);
W = (randn(M*N,nbrOfRealizations,K)+1i*randn(M*N,nbrOfRealizations,K));
H = zeros(M*N,nbrOfRealizations,K);

%Same channelGain_LoS and channelGain_NLoS for all realizations (at each setup) but phases shift of LoS are different at each
%coherence block


    
H_LoS = reshape(repmat(HMean_Withoutphase,nbrOfRealizations,1),M*N,nbrOfRealizations,K); 

for m = 1:M
    for k = 1:K
        
        R((m-1)*N+1:m*N,(m-1)*N+1:m*N,k) = R_AP(:,:,m,k);
        
    end
end


%Go through all UEs and apply the channel gains to the spatial
%correlation and mean matrices and introduce the phase shifts 
for k = 1:K
    
    Rsqrt = sqrtm(R(:,:,k));
    H(:,:,k) = sqrt(0.5)*Rsqrt*W(:,:,k) + H_LoS(:,:,k);
       
end
