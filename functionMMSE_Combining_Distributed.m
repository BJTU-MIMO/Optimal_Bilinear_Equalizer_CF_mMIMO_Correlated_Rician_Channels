function [V_MMSE_Combining] = functionMMSE_Combining_Distributed(Hhat,C_total,nbrOfRealizations,M,N,K,p)
%%=============================================================
%The file is used to design local MMSE combining vectors of the paper:
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
eyeN = eye(N);





%Diagonal matrix with transmit powers and its square root
Dp = diag(p);

V_MMSE_Combining = zeros(M*N,K,nbrOfRealizations);




%% Go through all channel realizations
for n = 1:nbrOfRealizations


    %Go through all APs
    for m = 1:M


        %Extract channel estimate realizations from all UEs to AP l
        Hhatallj = reshape(Hhat(1+(m-1)*N:m*N,n,:),[N K]);
        
   
        %Compute MMSE combining

        V_MMSE_Combining((m-1)*N+1:m*N,:,n) = ((Hhatallj*Dp*Hhatallj') + C_total(:,:,m) + eyeN)\(Hhatallj*Dp);

    end
end
        



V_MMSE_Combining = permute(V_MMSE_Combining,[1 3 2]);