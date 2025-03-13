function [SE_Centralized_UatF_Analytical,denominator_termkl,numerator_term,denominator_term2,Gbar_k,denominator_term1] = functionComputeSE_Centralized_UatF_Analytical(W,G_LoS_eff,Rhat,R,Phi,Pset,tau_c,tau_p,M,N,K,p)
%%=============================================================
%The file is used to compute the closed-form achievable SE with random phase-shifts for the centralized processing based on
%the UatF capacity bound (Theorem 1) of the paper:
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




% Define matrices
Gbar_k = zeros(M*N,M*N,K);
Gbar_kk_2 = zeros(M*N,M*N,K);

Rhat_blk = zeros(M*N,M*N,K);
Rbar = zeros(M*N,M*N,K);
R_blk = zeros(M*N,M*N,K);
Phi_blk = zeros(M*N,M*N,K);
Phi_cross_blk = zeros(M*N,M*N,K,K);

%% Compute the matrices applied in the computation of closed-form SE

for k = 1:K

    Gbar_kk_2(:,:,k) = G_LoS_eff(:,k)*G_LoS_eff(:,k)';

    for m = 1:M

        Rhat_blk((m-1)*N+1:m*N,(m-1)*N+1:m*N,k) = Rhat(:,:,m,k);
        R_blk((m-1)*N+1:m*N,(m-1)*N+1:m*N,k) = R(:,:,m,k);
        Phi_blk((m-1)*N+1:m*N,(m-1)*N+1:m*N,k) = Phi(:,:,m,k);

        Gbar_k((m-1)*N+1:m*N,(m-1)*N+1:m*N,k) = G_LoS_eff((m-1)*N+1:m*N,k)*G_LoS_eff((m-1)*N+1:m*N,k)';

    end

end



for k = 1:K
    
    Rbar(:,:,k) = Gbar_k(:,:,k) + Rhat_blk(:,:,k);

    for l = 1:K
        if any(l == Pset(:,k))

            Phi_cross_blk(:,:,l,k) = R_blk(:,:,l)/Phi_blk(:,:,k)*R_blk(:,:,k);

        end
    end
end




for k = 1:K

    numerator_term(k) = p(k)*abs(trace(W(:,:,k)'*Rbar(:,:,k)))^2;

    for l = 1:K

        denominator_termkl(k,l) = p(l)*trace(W(:,:,k)'*Gbar_k(:,:,l)*W(:,:,k)*Gbar_k(:,:,k)) + p(l)*trace(W(:,:,k)'*R_blk(:,:,l)*W(:,:,k)*Gbar_k(:,:,k))...
            + p(l)*trace(W(:,:,k)'*Gbar_k(:,:,l)*W(:,:,k)*Rhat_blk(:,:,k)) + p(l)*trace(W(:,:,k)'*R_blk(:,:,l)*W(:,:,k)*Rhat_blk(:,:,k));


        if any(l==Pset(:,k))

            denominator_termkl(k,l) = denominator_termkl(k,l) + p(l)*p(k)*p(l)*tau_p^2*abs(trace(W(:,:,k)'*Phi_cross_blk(:,:,l,k)))^2;


        end


        if l == k


            denominator_termkl(k,l) = denominator_termkl(k,l) + p(l)*sqrt(p(k)*p(l))*tau_p*trace(W(:,:,k)'*Gbar_k(:,:,k))*trace(Phi_cross_blk(:,:,l,k)'*W(:,:,k))...
                + p(l)*sqrt(p(k)*p(l))*tau_p*trace(W(:,:,k)'*Phi_cross_blk(:,:,l,k))*trace(Gbar_k(:,:,k)*W(:,:,k)) - p(l)*trace(W(:,:,k)'*Gbar_k(:,:,k)*W(:,:,k)*Gbar_k(:,:,k));



            for m1 = 1:M
                for m2 = 1:M

                    gm1 = G_LoS_eff((m1-1)*N+1:m1*N,k);
                    gm2 = G_LoS_eff((m2-1)*N+1:m2*N,k);
                    WH = W(:,:,k)';

                    if m1~=m2

                        denominator_termkl(k,l)  = denominator_termkl(k,l)  + p(l)*(gm1'*WH((m1-1)*N+1:m1*N,(m1-1)*N+1:m1*N)'*gm1*gm2'*WH((m2-1)*N+1:m2*N,(m2-1)*N+1:m2*N)*gm2...
                            + gm1'*WH((m2-1)*N+1:m2*N,(m1-1)*N+1:m1*N)'*gm2*gm2'*WH((m2-1)*N+1:m2*N,(m1-1)*N+1:m1*N)*gm1);


                    end

                    if m1 == m2

                        denominator_termkl(k,l) = denominator_termkl(k,l) + p(l)*(gm1'*WH((m1-1)*N+1:m1*N,(m1-1)*N+1:m1*N)'*gm1*gm1'*WH((m1-1)*N+1:m1*N,(m1-1)*N+1:m1*N)*gm1);

                    end

                end
            end
            


        end




    end

    denominator_term2(k) = trace(W(:,:,k)*Rbar(:,:,k)*W(:,:,k)');


end

for k = 1:K
    
    denominator_term1 = sum(denominator_termkl,2);

    denominator_k = denominator_term1(k) - numerator_term(k) + denominator_term2(k);

    SE_Centralized_UatF_Analytical(k) = prelogFactor*real(log2(1+ numerator_term(k)/denominator_k));

end






