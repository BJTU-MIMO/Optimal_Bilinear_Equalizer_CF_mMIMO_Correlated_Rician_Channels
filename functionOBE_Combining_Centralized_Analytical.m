function [V_OBE_Combining_Centralized,W_OBE_matrix] = functionOBE_Combining_Centralized_Analytical(GLoS_Withoutphase,Ghat_MMSE,Rhat,Phi,R,Pset,M,N,K,p,tau_p,nbrOfRealizations)
%%=============================================================
%The file is used to design centralized OBE combining vectors with random phase-shifts based on closed-form OBE matrices in Theorem 3 of the paper:
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



G_LoS_eff = GLoS_Withoutphase;

% Define matrices
Gbar_k = zeros(M*N,M*N,K);
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
%% Centralized OBE combining design


for k = 1:K

    Gamma = zeros((M*N)^2,(M*N)^2);
    Upsilon = zeros((M*N)^2,(M*N)^2);

    for e = 1:M*N
        for f = 1:M*N

            m1 = floor((f-1)/N)+1;
            m2 = floor((e-1)/N)+1;
            n1 = mod(f,N);
            n2 = mod(e,N);

            if n1 == 0

                n1 = N;

            end

            if n2 == 0

                n2 = N;
                
            end

            for s = 1:M*N
                for t = 1:M*N

                    m3 = floor((s-1)/N)+1;
                    m4 = floor((t-1)/N)+1;
                    n3 = mod(s,N);
                    n4 = mod(t,N);

                    if n3 == 0

                        n3 = N;

                    end

                    if n4 == 0

                        n4 = N;

                    end


                    if m1 == m2 && m3 == m4 && m1~=m3

                        Upsilon((e-1)*M*N+s,(f-1)*M*N+t) = G_LoS_eff((m1-1)*N+n1,k)*G_LoS_eff((m2-1)*N+n2,k)'*G_LoS_eff((m3-1)*N+n3,k)*G_LoS_eff((m4-1)*N+n4,k)';

                    else

                        if m1 == m4 && m2 == m3 && m1~=m2

                            Upsilon((e-1)*M*N+s,(f-1)*M*N+t) = G_LoS_eff((m1-1)*N+n1,k)*G_LoS_eff((m2-1)*N+n2,k)'*G_LoS_eff((m3-1)*N+n3,k)*G_LoS_eff((m4-1)*N+n4,k)';

                        else

                            if m1 == m2 && m3 == m4 && m1 == m3

                                Upsilon((e-1)*M*N+s,(f-1)*M*N+t) = G_LoS_eff((m1-1)*N+n1,k)*G_LoS_eff((m2-1)*N+n2,k)'*G_LoS_eff((m3-1)*N+n3,k)*G_LoS_eff((m4-1)*N+n4,k)';

                            end
                        end
                    end

                end
            end
        end
    end

    for l = 1:K

        Gamma = Gamma + p(l)*kron(Gbar_k(:,:,k).',Gbar_k(:,:,l)) + p(l)*kron(Gbar_k(:,:,k).',R_blk(:,:,l)) + p(l)*kron(Rhat_blk(:,:,k).',Gbar_k(:,:,l))...
            + p(l)*kron(Rhat_blk(:,:,k).',R_blk(:,:,l));

        if any(l == Pset(:,k))

            rwave_lk = Phi_cross_blk(:,:,l,k);
            Gamma = Gamma + p(l)*p(k)*p(l)*tau_p^2*rwave_lk(:)*rwave_lk(:)';

        end

        if l == k

            gbar_kk = Gbar_k(:,:,k);
            rwave_lk = Phi_cross_blk(:,:,l,k);

            Gamma = Gamma + p(l)*sqrt(p(k)*p(l))*tau_p*gbar_kk(:)*rwave_lk(:)' + p(l)*sqrt(p(k)*p(l))*tau_p*rwave_lk(:)*gbar_kk(:)'...
                + p(k)*Upsilon - p(k)*kron(Gbar_k(:,:,k).',Gbar_k(:,:,k));


        end

    end

    rbar_k = Rbar(:,:,k);

    Gamma = Gamma - p(k)*rbar_k(:)*rbar_k(:)' + kron(Rbar(:,:,k).',eye(M*N));
    



    W_OBE_vector = Gamma\rbar_k(:);

    W_OBE_matrix(:,:,k) = reshape(W_OBE_vector,[M*N,M*N]);

    V_OBE_Combining_Centralized(:,:,k) = W_OBE_matrix(:,:,k)*Ghat_MMSE(:,:,k);

    clear Gamma W_OBE_vector Upsilon



end



  




