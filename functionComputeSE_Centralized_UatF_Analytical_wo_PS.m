function [SE_Centralized_UatF_Analytical] = functionComputeSE_Centralized_UatF_Analytical_wo_PS(W,G_LoS_eff,Rhat,R,Phi,Pset,tau_c,tau_p,M,N,K,p)
%%=============================================================
%The file is used to compute the closed-form achievable SE without random phase-shifts for the centralized processing based on
%the UatF capacity bound of the paper:
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


%Compute the prelog factor
prelogFactor = (1-tau_p/tau_c);

%Prepare to store simulation results
SE_Centralized_UatF_Analytical = zeros(K,1);



numerator_term = zeros(K,1);
denominator_termkl = zeros(K,K);
denominator_term2 = zeros(K,1);



G_LoS_eff = reshape(G_LoS_eff(:,1,:),M*N,K);

% Define matrices
Gbar_kl = zeros(M*N,M*N,K,K);
Rhat_blk = zeros(M*N,M*N,K);
Rbar = zeros(M*N,M*N,K);
R_blk = zeros(M*N,M*N,K);
Phi_blk = zeros(M*N,M*N,K);
Phi_cross_blk = zeros(M*N,M*N,K,K);

%% Compute the matrices applied in the computation of closed-form SE

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




for k = 1:K

    numerator_term(k) = p(k)*abs(trace(W(:,:,k)'*Rbar(:,:,k)))^2;

    for l = 1:K

        denominator_termkl(k,l) = denominator_termkl(k,l) + p(l)*trace(W(:,:,k)'*Gbar_kl(:,:,l,l)*W(:,:,k)*Gbar_kl(:,:,k,k)) + p(l)*trace(W(:,:,k)'*R_blk(:,:,l)*W(:,:,k)*Gbar_kl(:,:,k,k))...
            + p(l)*trace(W(:,:,k)'*Gbar_kl(:,:,l,l)*W(:,:,k)*Rhat_blk(:,:,k)) + p(l)*trace(W(:,:,k)'*R_blk(:,:,l)*W(:,:,k)*Rhat_blk(:,:,k));

        if any(l==Pset(:,k))

            denominator_termkl(k,l) = denominator_termkl(k,l) + p(l)*sqrt(p(k)*p(l))*tau_p*trace(W(:,:,k)'*Gbar_kl(:,:,l,k))*trace(Phi_cross_blk(:,:,l,k)'*W(:,:,k))...
                + p(l)*sqrt(p(k)*p(l))*tau_p*trace(W(:,:,k)'*Phi_cross_blk(:,:,l,k))*trace(Gbar_kl(:,:,k,l)*W(:,:,k))...
                + p(l)*p(k)*p(l)*tau_p^2*abs(trace(W(:,:,k)'*Phi_cross_blk(:,:,l,k)))^2;

        end

    end

    denominator_term2(k) = trace(W(:,:,k)*Rbar(:,:,k)*W(:,:,k)');


end

for k = 1:K
    
    denominator_term1 = sum(denominator_termkl,2);

    denominator_k = denominator_term1(k) - numerator_term(k) + denominator_term2(k);

    SE_Centralized_UatF_Analytical(k) = prelogFactor*real(log2(1+ numerator_term(k)/denominator_k));

end






