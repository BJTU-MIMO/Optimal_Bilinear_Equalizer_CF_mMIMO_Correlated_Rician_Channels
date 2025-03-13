function [V_OBE_Combining_Centralized,W_OBE_matrix] = functionOBE_Combining_Centralized_Analytical_wo_PS(G_LoS_eff,Ghat_MMSE,Rhat,Phi,R,Pset,M,N,K,p,tau_p,nbrOfRealizations)
%%=============================================================
%The file is used to design centralized OBE combining vectors without random phase-shifts based on closed-form OBE matrices of the paper:
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



G_LoS_eff = reshape(G_LoS_eff(:,1,:),M*N,K);

% Define matrices
Gbar_kl = zeros(M*N,M*N,K,K);
Rhat_blk = zeros(M*N,M*N,K);
Rbar = zeros(M*N,M*N,K);
R_blk = zeros(M*N,M*N,K);
Phi_blk = zeros(M*N,M*N,K);
Phi_cross_blk = zeros(M*N,M*N,K,K);
V_OBE_Combining_Centralized = zeros(M*N,nbrOfRealizations,K);
W_OBE_matrix = zeros(M*N,M*N,K);

%% Compute the matrices applied in the OBE combining design

for k = 1:K

    for m = 1:M

        Rhat_blk((m-1)*N+1:m*N,(m-1)*N+1:m*N,k) = Rhat(:,:,m,k);
        R_blk((m-1)*N+1:m*N,(m-1)*N+1:m*N,k) = R(:,:,m,k);
        Phi_blk((m-1)*N+1:m*N,(m-1)*N+1:m*N,k) = Phi(:,:,m,k);

    end

    Rbar(:,:,k) = G_LoS_eff(:,k)*G_LoS_eff(:,k)' + Rhat_blk(:,:,k);


        for l = 1:K

            Gbar_kl(:,:,k,l) = G_LoS_eff(:,k)*G_LoS_eff(:,l)';

        end
end

for k = 1:K
    for l = 1:K
        if any(l == Pset(:,k))

            Phi_cross_blk(:,:,l,k) = R_blk(:,:,l)/Phi_blk(:,:,k)*R_blk(:,:,k);

        end
    end
end
%% Centralized OBE combining design


for k = 1:K

    Gamma = zeros((M*N)^2,(M*N)^2);


    for l = 1:K

    Gamma = Gamma + p(l)*kron(Gbar_kl(:,:,k,k).',Gbar_kl(:,:,l,l)) + p(l)*kron(Gbar_kl(:,:,k,k).',R_blk(:,:,l)) + p(l)*kron(Rhat_blk(:,:,k).',Gbar_kl(:,:,l,l))...
        + p(l)*kron(Rhat_blk(:,:,k).',R_blk(:,:,l));

    if any(l == Pset(:,k))

        gbar_lk = Gbar_kl(:,:,l,k);
        rwave_lk = Phi_cross_blk(:,:,l,k);


        Gamma = Gamma + p(l)*sqrt(p(k)*p(l))*tau_p*gbar_lk(:)*rwave_lk(:)' + p(l)*sqrt(p(k)*p(l))*tau_p*rwave_lk(:)*gbar_lk(:)'...
            + p(l)*p(k)*p(l)*tau_p^2*rwave_lk(:)*rwave_lk(:)';

    end
    end

    rbar_k = Rbar(:,:,k);

    Gamma = Gamma - p(k)*rbar_k(:)*rbar_k(:)' + kron(Rbar(:,:,k).',eye(M*N));



    W_OBE_vector = Gamma\rbar_k(:);

    W_OBE_matrix(:,:,k) = reshape(W_OBE_vector,[M*N,M*N]);

    V_OBE_Combining_Centralized(:,:,k) = W_OBE_matrix(:,:,k)*Ghat_MMSE(:,:,k);

    clear Gamma W_OBE_vector 



end


