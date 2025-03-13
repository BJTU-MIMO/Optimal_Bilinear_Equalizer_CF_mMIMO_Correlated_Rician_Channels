function [Hhat_LS] = functionChannelEstimates_LS(H,nbrOfRealizations,M,K,N,tau_p,p,Pset)
%%=============================================================
%The file is used to generate the LS channel estimates of the paper:
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


%Prepare to store LS channel estimates
Hhat_LS = zeros(M*N,nbrOfRealizations,K);

%Generate realizations of normalized noise
Np = sqrt(0.5)*(randn(N,nbrOfRealizations,M,K) + 1i*randn(N,nbrOfRealizations,M,K));





for m = 1:M
    for k = 1:K
        
        yp = zeros(N,nbrOfRealizations);
        inds = Pset(:,k); 
        
        for z = 1:length(inds)
            
            yp = yp + sqrt(p(inds(z)))*tau_p*H((m-1)*N+1:m*N,:,inds(z));
            
        end
        
        yp = yp + sqrt(tau_p)*Np(:,:,m,k);
        
        for z = 1:length(inds) 
            
            Hhat_LS((m-1)*N+1:m*N,:,inds(z)) = 1/(sqrt(p(inds(z)))*tau_p)*yp;
                
        end
    end
end

       