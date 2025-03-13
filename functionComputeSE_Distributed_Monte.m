function [SE_LSFD,SE_Distributed,A_LSFD] = functionComputeSE_Distributed_Monte(H,V_Combining,tau_c,tau_p,nbrOfRealizations,N,K,M,p)
%%=============================================================
%The file is used to compute the achievable SE for the distributed processing based on Monte-Carlo methods of the paper:
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

%If no specific Level 1 transmit powers are provided, use the same as for
%the other levels
if nargin<12
    p1 = p;
end


%Store identity matrices of different sizes


%Compute the prelog factor
prelogFactor = (1-tau_p/tau_c);

%Prepare to store simulation results
SE_LSFD = zeros(K,1);
SE_Distributed = zeros(K,1);

%Compute sum of all estimation error correlation matrices at every BS





%Diagonal matrix with transmit powers and its square root
Dp12 = diag(sqrt(p));


%Prepare to save simulation results

signal = zeros(M,K);
scaling = zeros(M,K);
G = zeros(M,M,K);
A = zeros(M,M,K);


%% Go through all channel realizations
for n = 1:nbrOfRealizations
    

    gp = zeros(M,K,K);
    
    
    %Go through all APs
    for m = 1:M
        
        %Extract channel realizations from all UEs to AP l
        Hallj = reshape(H(1+(m-1)*N:m*N,n,:),[N K]);

        V = reshape(V_Combining(1+(m-1)*N:m*N,n,:),[N K]);

        %Go through all UEs
        for k = 1:K
            

            v = V(:,k); %Extract combining vector

            signal(m,k) = signal(m,k) + (v'*Hallj(:,k))/nbrOfRealizations;
            gp(m,:,k) = gp(m,:,k) + (v'*Hallj)*Dp12;
            scaling(m,k) = scaling(m,k) + norm(v).^2/nbrOfRealizations;
            
           
        end
        
    end
    
    for k = 1:K
        
        G(:,:,k) = G(:,:,k) + gp(:,:,k)*gp(:,:,k)'/nbrOfRealizations;
        
    end
    
end

A_LSFD = zeros(M,K);

for k = 1:K
    

    b = signal(:,k);
    A(:,:,k) = G(:,:,k) + diag(scaling(:,k)) - p(k)*(b*b');
    A_LSFD(:,k) = A(:,:,k)\b;

    SE_LSFD(k) = prelogFactor*real(log2(1+p(k)*b'*(A(:,:,k)\b)));

    

    SE_Distributed(k) = prelogFactor*real(log2(1+p(k)*abs(mean(b)).^2 / mean(mean(A(:,:,k)))));


    

end
