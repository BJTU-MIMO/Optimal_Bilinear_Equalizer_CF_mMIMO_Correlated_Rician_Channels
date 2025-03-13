function [V_OBE_Combining_Distributed_Local,W_OBE_matrix] = functionOBE_Combining_Distributed_Local(G_LoS_eff,Ghat_MMSE,Rhat,Phi,R,Pset,M,N,K,p,tau_p,nbrOfRealizations)
%%=============================================================
%The file is used to design DL-OBE combining vectors with random phase-shifts based on closed-form OBE matrices in Theorem 5 of the paper:
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



V_OBE_Combining_Distributed_Local = zeros(M*N,nbrOfRealizations,K);

W_OBE_matrix = zeros(N,N,M,K);

%% Distributed OBE combining design at each AP based on local information


for m = 1:M
    for k = 1:K

        Gamma_mk = zeros((N)^2,(N)^2);

        for l = 1:K


            Gbar_mkk = G_LoS_eff((m-1)*N+1:m*N,k)*G_LoS_eff((m-1)*N+1:m*N,k)';
            Gbar_mll = G_LoS_eff((m-1)*N+1:m*N,l)*G_LoS_eff((m-1)*N+1:m*N,l)';

            Gamma_mk = Gamma_mk + p(l)*kron(Gbar_mkk.',Gbar_mll) + p(l)*kron(Gbar_mkk.',R(:,:,m,l)) + p(l)*kron(Rhat(:,:,m,k).',Gbar_mll) + p(l)*kron(Rhat(:,:,m,k).',R(:,:,m,l));


            if any(l == Pset(:,k))

                Phi_cross_mlk = R(:,:,m,l)/Phi(:,:,m,k)*R(:,:,m,k);

  
                rwave_mlk = Phi_cross_mlk(:);


                Gamma_mk = Gamma_mk + p(l)*p(k)*p(l)*tau_p^2*(rwave_mlk*rwave_mlk');



                if l == k

                    gbar_mkk = Gbar_mkk;
       

                    Gamma_mk = Gamma_mk + p(l)*sqrt(p(k)*p(l))*tau_p*gbar_mkk(:)*rwave_mlk(:)' + p(l)*sqrt(p(k)*p(l))*tau_p*rwave_mlk(:)*gbar_mkk(:)';
   

                end



            end

        end


        Rbar_mk = G_LoS_eff((m-1)*N+1:m*N,k)*G_LoS_eff((m-1)*N+1:m*N,k)' + Rhat(:,:,m,k);

        rbar_mk = Rbar_mk(:);

        Gamma_mk = Gamma_mk - p(k)*(rbar_mk*rbar_mk') + kron(Rbar_mk.',eye(N));


        W_OBE_vector_mk = Gamma_mk\rbar_mk;
        W_OBE_matrix_mk = reshape(W_OBE_vector_mk,[N,N]);

        W_OBE_matrix(:,:,m,k) = W_OBE_matrix_mk;

        V_OBE_Combining_Distributed_Local((m-1)*N+1:m*N,:,k) = W_OBE_matrix_mk*Ghat_MMSE((m-1)*N+1:m*N,:,k);

        clear Gamma_mk W_OBE_vector_mk W_OBE_matrix_mk
        
    end



end

