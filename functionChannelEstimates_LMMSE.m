function [Hhat_LMMSE] = functionChannelEstimates_LMMSE(R,HMean_Withoutphase,H,nbrOfRealizations,M,K,N,tau_p,p,Pset)
%%=============================================================
%The file is used to generate the LMMSE channel estimates of the paper:
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



%Prepare to store LMMSE channel estimates
Hhat_LMMSE = zeros(M*N,nbrOfRealizations,K);

%Generate realizations of normalized noise
Np = sqrt(0.5)*(randn(N,nbrOfRealizations,M,K) + 1i*randn(N,nbrOfRealizations,M,K));

%Store identity matrix of size N x N
eyeN = eye(N);

Lk = zeros(N,N,M,K);
for m = 1:M
    for k = 1:K
        
         Lk(:,:,m,k) = HMean_Withoutphase((m-1)*N+1:m*N,k)*(HMean_Withoutphase((m-1)*N+1:m*N,k))';
         
    end
end
 

Rp = R + Lk;


for m = 1:M
    for k = 1:K
        
        yp = zeros(N,nbrOfRealizations);
        PsiInv_LMMSE = zeros(N,N);
        inds = Pset(:,k); 
        
        for z = 1:length(inds)
            
            yp = yp + sqrt(p(inds(z)))*tau_p*H((m-1)*N+1:m*N,:,inds(z));
            PsiInv_LMMSE = PsiInv_LMMSE + p(inds(z))*tau_p*Rp(:,:,m,inds(z));
            
        end
        
        yp = yp + sqrt(tau_p)*Np(:,:,m,k);
        PsiInv_LMMSE = PsiInv_LMMSE + eyeN;
        
        for z = 1:length(inds) 
            
            RPsi = Rp(:,:,m,inds(z))/PsiInv_LMMSE;
            Hhat_LMMSE((m-1)*N+1:m*N,:,inds(z)) = sqrt(p(inds(z)))*RPsi*yp;
                
        end
    end
end

       