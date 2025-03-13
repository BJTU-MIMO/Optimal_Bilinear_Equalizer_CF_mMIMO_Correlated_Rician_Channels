function [SE_Centralized] = functionComputeSE_Centralized(Hhat,V_Combining,C_tot_blk,tau_c,tau_p,nbrOfRealizations,M,N,K,p)
%%=============================================================
%The file is used to compute the achievable SE for the centralized processing by Monte-Carlo methods based on
%the standard capacity lower bound of the paper:
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

%If only one transmit power is provided, use the same for all the UEs
if length(p) == 1
   p = p*ones(K,1);
end


%Compute the prelog factor
prelogFactor = (1-tau_p/tau_c);

%Prepare to store simulation results
SE_Centralized = zeros(K,1);

eyeMN = eye(M*N);


%Diagonal matrix with transmit powers and its square root
Dp12 = diag(sqrt(p));



%% Go through all channel realizations
for n = 1:nbrOfRealizations
    
    
    
    %Extract channel estimate and combining vector realizations from all UEs to all APs
    Hhatallj = reshape(Hhat(:,n,:),[M*N K]);
   
    V_n = reshape(V_Combining(:,n,:),[M*N K]);
    

    %Go through all UEs
    for k = 1:K
        
        v = V_n(:,k); %Extract combining vector
        
        %Compute numerator and denominator of instantaneous SINR at Level 4
        numerator = p(k)*abs(v'*Hhatallj(:,k))^2;
        denominator = norm(v'*Hhatallj*Dp12)^2 + v'*(C_tot_blk+eyeMN)*v - numerator;
        
        %Compute instantaneous SE for one channel realization
        SE_Centralized(k) = SE_Centralized(k) + prelogFactor*real(log2(1+numerator/denominator))/nbrOfRealizations;
        
    end
    
  
end



