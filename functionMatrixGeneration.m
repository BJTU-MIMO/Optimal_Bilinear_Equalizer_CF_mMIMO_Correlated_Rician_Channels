function [RHat,Phi,C_MMSE,C_total,C_total_blk] = functionMatrixGeneration(R,p,M,K,N,tau_p,Pset)
%%=============================================================
%The file is used to generate the matrices based on the MMSE channel estimator used in the next subsequent calculation of the paper:
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


%If only one transmit power is provided, use the same for all the UEs
if length(p) == 1
   p = p*ones(K,1);
end

% Prepare to store the result
Phi = zeros(N,N,M,K); 
RHat = zeros(N,N,M,K);
C_MMSE = zeros(N,N,M,K);
C_total = zeros(N,N,M);
C_total_blk = zeros(M*N,M*N);

% Go through all sub-arrays          
for m = 1:M
    
    % Go through all UEs
    for k = 1:K
        
        % Compute the UEs indexes that use the same pilot as UE k
        inds = Pset(:,k);
        PsiInv = zeros(N,N);
        
        % Go through all UEs that use the same pilot as UE k 
        for z = 1:length(inds)   
            
            PsiInv = PsiInv + p(inds(z))*tau_p*R(:,:,m,inds(z));

        end
            PsiInv = PsiInv + eye(N);

            
            for z = 1:length(inds)
                
                Phi(:,:,m,inds(z)) = PsiInv;
            
            end
            
            RHat(:,:,m,k) = p(k)*tau_p*R(:,:,m,k)/PsiInv*R(:,:,m,k);

            
    end
end

% Generate estimation error correlation matrices
for k = 1:K
    
    C_MMSE(:,:,:,k) = R(:,:,:,k) - RHat(:,:,:,k);
    C_total = C_total + p(k)*C_MMSE(:,:,:,k); 
   
end

for m = 1:M

    C_total_blk((m-1)*N+1:m*N,(m-1)*N+1:m*N) = C_total(:,:,m);

end

        
                        

            
