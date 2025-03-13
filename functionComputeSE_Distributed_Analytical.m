function [SE] = functionComputeSE_Distributed_Analytical(W,GLoS_Withoutphase,Rhat,R,Phi,tau_c,tau_p,Pset,N,K,M,p)
%%=============================================================
%The file is used to compute the closed-form achievable SE with random phase-shifts for the distributed processing of the paper:
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






SE = zeros(K,1);
G_LoS_eff = GLoS_Withoutphase;

% Define matrices
Gbar_mkk = zeros(N,N,M,K);
B_mlk = zeros(N,N,M,K,K);
Rbar_mk = zeros(N,N,M,K);
Phi_cross_mlk = zeros(N,N,M,K,K);
%% Compute the matrices applied in the OBE combining design

for k = 1:K
    for m = 1:M

        Gbar_mkk(:,:,m,k) = G_LoS_eff((m-1)*N+1:m*N,k)*G_LoS_eff((m-1)*N+1:m*N,k)';
        Rbar_mk(:,:,m,k) = Gbar_mkk(:,:,m,k) + Rhat(:,:,m,k);

        for l = 1:K



            B_mlk(:,:,m,l,k) = sqrt(p(k)*p(l))*tau_p*R(:,:,m,l)/Phi(:,:,m,k)*R(:,:,m,k);

            if l == k

                B_mlk(:,:,m,l,k) = Gbar_mkk(:,:,m,k) + sqrt(p(k)*p(l))*tau_p*R(:,:,m,l)/Phi(:,:,m,k)*R(:,:,m,k);

            end


            Phi_cross_mlk(:,:,m,l,k) = R(:,:,m,l)/Phi(:,:,m,k)*R(:,:,m,k);


        end

    end
end




numerator = zeros(K,1);
mu_mkl = zeros(K,1);
gamma = zeros(K,K,M);
denominator3 = zeros(K,1);


for k = 1:K
    for m = 1:M

        numerator(k) = numerator(k) + trace(W(:,:,m,k)'*Rbar_mk(:,:,m,k));


        denominator3(k) = denominator3(k) + trace(W(:,:,m,k)'*W(:,:,m,k)*Gbar_mkk(:,:,m,k)) + trace(W(:,:,m,k)'*W(:,:,m,k)*Rhat(:,:,m,k));

        for l = 1:K

            mu_mkl(k) = mu_mkl(k) + p(l)*trace(W(:,:,m,k)'*Gbar_mkk(:,:,m,l)*W(:,:,m,k)*Gbar_mkk(:,:,m,k)) + p(l)*trace(W(:,:,m,k)'*R(:,:,m,l)*W(:,:,m,k)*Gbar_mkk(:,:,m,k))...
                + p(l)*trace(W(:,:,m,k)'*Gbar_mkk(:,:,m,l)*W(:,:,m,k)*Rhat(:,:,m,k)) + p(l)*trace(W(:,:,m,k)'*R(:,:,m,l)*W(:,:,m,k)*Rhat(:,:,m,k));



            if any(l == Pset(:,k))

                mu_mkl(k) = mu_mkl(k) + p(l)*p(k)*p(l)*tau_p^2*abs(trace(W(:,:,m,k)'*Phi_cross_mlk(:,:,m,l,k)))^2;

                gamma(k,l,m) = trace(W(:,:,m,k)'*B_mlk(:,:,m,l,k));

                if l == k

                    mu_mkl(k) = mu_mkl(k) + p(l)*sqrt(p(k)*p(l))*tau_p*trace(W(:,:,m,k)'*Gbar_mkk(:,:,m,k))*trace(W(:,:,m,k)*Phi_cross_mlk(:,:,m,k,l)) + p(l)*sqrt(p(k)*p(l))*tau_p*trace(W(:,:,m,k)'*Phi_cross_mlk(:,:,m,l,k))*trace(W(:,:,m,k)*Gbar_mkk(:,:,m,k));

                end
                 

            end

        end
    end
end


denominator2_1 = 0;
denominator2_2 = 0;
denominator2 = zeros(K,1);

for k = 1:K

    for l = 1:K

        for m = 1:M

            denominator2_1 = denominator2_1 + gamma(k,l,m);
            denominator2_2 = denominator2_2 + abs(gamma(k,l,m))^2;

        end

        denominator2_1 = abs(denominator2_1)^2;

        denominator2(k) = denominator2(k) + p(l)*denominator2_1 - p(l)*denominator2_2;

        denominator2_1 = 0;
        denominator2_2 = 0;

    end
end


for k = 1:K


    Numerator =  p(k)*numerator(k)*numerator(k)';

    Denominator = mu_mkl(k) - p(k)*numerator(k)*numerator(k)' + denominator2(k) + denominator3(k);

    SE(k) = prelogFactor*real(log2(1+ Numerator/Denominator));

end


    



