function [V_OBE_Combining_Distributed,W_OBE_matrix] = functionOBE_Combining_Distributed(G_LoS_eff,Ghat_MMSE,Rhat,Phi,R,Pset,M,N,K,p,tau_p,nbrOfRealizations)
%%=============================================================
%The file is used to design DG-OBE combining vectors with random phase-shifts based on closed-form OBE matrices in Theorem 4 of the paper:
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



% Define matrices

Gbar_mkk = zeros(N,N,M,K);
Bbar_lk_blk = zeros(M*N^2,M*N^2,K,K);
Bhat_lk = zeros(M*N^2,M*N^2,K,K);
B_mlk = zeros(N,N,M,K,K);
Rbar_mk = zeros(N,N,M,K);
Phi_cross_mlk = zeros(N,N,M,K,K);
rbar_k = zeros(M*N^2,K);


Lambda_kl_1 = zeros(M*N^2,M*N^2,K,K);
Lambda_kl_2 = zeros(M*N^2,M*N^2,K,K);
Lambda_kl_3 = zeros(M*N^2,M*N^2,K,K);
Lambda_k_4 = zeros(M*N^2,M*N^2,K);

W_OBE_matrix = zeros(N,N,M,K);

V_OBE_Combining_Distributed = zeros(M*N,nbrOfRealizations,K);


%% Compute the matrices applied in the OBE combining design

for k = 1:K
    for m = 1:M

        Gbar_mkk(:,:,m,k) = G_LoS_eff((m-1)*N+1:m*N,k)*G_LoS_eff((m-1)*N+1:m*N,k)';

        for l = 1:K

            Rbar_mk(:,:,m,k) = Gbar_mkk(:,:,m,k) + Rhat(:,:,m,k);

            Lambda_k_4((m-1)*N^2+1:m*N^2,(m-1)*N^2+1:m*N^2,k) = kron(Gbar_mkk(:,:,m,k).',eye(N)) + kron(Rhat(:,:,m,k).',eye(N));


            B_mlk(:,:,m,l,k) = sqrt(p(k)*p(l))*tau_p*R(:,:,m,l)/Phi(:,:,m,k)*R(:,:,m,k);

            if l == k

                B_mlk(:,:,m,l,k) = Gbar_mkk(:,:,m,k) + sqrt(p(k)*p(l))*tau_p*R(:,:,m,l)/Phi(:,:,m,k)*R(:,:,m,k);

            end

            b_mlk_vec = B_mlk(:,:,m,l,k);

            b_mlk_vec = b_mlk_vec(:);

            Bbar_lk_blk((m-1)*N^2+1:m*N^2,(m-1)*N^2+1:m*N^2,l,k) = b_mlk_vec*b_mlk_vec';

            Phi_cross_mlk(:,:,m,l,k) = R(:,:,m,l)/Phi(:,:,m,k)*R(:,:,m,k);


        end

        rbar_mk = Rbar_mk(:,:,m,k);
        rbar_mk = rbar_mk(:);
        rbar_k((m-1)*N^2+1:m*N^2,k) = rbar_mk;

    end
end




for k = 1:K
    for l = 1:K

        for m = 1:M


            for mm = 1:M




                b_mlk = B_mlk(:,:,m,l,k);
                b_mlk = b_mlk(:);

                b_mmlk = B_mlk(:,:,mm,l,k);
                b_mmlk = b_mmlk(:);

                Bhat_lk((m-1)*N^2+1:m*N^2,(mm-1)*N^2+1:mm*N^2,l,k) = b_mlk*b_mmlk';

            end

            Lambda_kl_1((m-1)*N^2+1:m*N^2,(m-1)*N^2+1:m*N^2,k,l) = p(l)*kron(Gbar_mkk(:,:,m,k).',Gbar_mkk(:,:,m,l)) + p(l)*kron(Gbar_mkk(:,:,m,k).',R(:,:,m,l))...
                + p(l)*kron(Rhat(:,:,m,k).',Gbar_mkk(:,:,m,l)) + p(l)*kron(Rhat(:,:,m,k).',R(:,:,m,l));


            if any(l == Pset(:,k))

                rwave_mlk = Phi_cross_mlk(:,:,m,l,k);
                rwave_mlk = rwave_mlk(:);
                Lambda_kl_2((m-1)*N^2+1:m*N^2,(m-1)*N^2+1:m*N^2,k,l) = p(l)*p(k)*p(l)*tau_p^2*(rwave_mlk*rwave_mlk');

                gbar_mkk = Gbar_mkk(:,:,m,k);
                gbar_mkk = gbar_mkk(:);

                if l == k

                    Lambda_kl_2((m-1)*N^2+1:m*N^2,(m-1)*N^2+1:m*N^2,k,l) = Lambda_kl_2((m-1)*N^2+1:m*N^2,(m-1)*N^2+1:m*N^2,k,l) + p(l)*(sqrt(p(k)*p(l))*tau_p*gbar_mkk*rwave_mlk' + sqrt(p(k)*p(l))*tau_p*rwave_mlk*gbar_mkk');
                    
                end


            end




            
        end


        Lambda_kl_3(:,:,k,l) = p(l)*Bhat_lk(:,:,l,k) - p(l)*Bbar_lk_blk(:,:,l,k);

    end
end




%% Distributed OBE combining design

for k = 1:K

    Lambda = zeros(M*N^2,M*N^2);

    for l = 1:K

        Lambda = Lambda + Lambda_kl_1(:,:,k,l);

        if any(l == Pset(:,k))

            Lambda = Lambda + Lambda_kl_2(:,:,k,l) + Lambda_kl_3(:,:,k,l);

        end

    end
    
    Lambda = Lambda - p(k)*(rbar_k(:,k)*rbar_k(:,k)') + Lambda_k_4(:,:,k);

        W_OBE_vector = Lambda\rbar_k(:,k);
        

    for m = 1:M

        W_OBE_vector_m = W_OBE_vector((m-1)*N^2+1:m*N^2);

        W_OBE_matrix_m = reshape(W_OBE_vector_m,[N,N]);

        W_OBE_matrix(:,:,m,k) = W_OBE_matrix_m;

        V_OBE_Combining_Distributed((m-1)*N+1:m*N,:,k) = W_OBE_matrix(:,:,m,k)*Ghat_MMSE((m-1)*N+1:m*N,:,k);

    end

    clear Gamma W_OBE_vector W_OBE_matrix_m 

end

clear Gbar_mkk Bbar_lk_blk Bhat_lk B_mlk Rbar_mk Phi_cross_mlk Gbar_lk_2 Gbarbar_lk_blk
