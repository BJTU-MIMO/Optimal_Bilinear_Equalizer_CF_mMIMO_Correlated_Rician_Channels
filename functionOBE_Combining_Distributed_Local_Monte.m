function [V_OBE_Combining_Distributed_Local_Monte,W_OBE_matrix] = functionOBE_Combining_Distributed_Local_Monte(G,Ghat_MMSE,M,N,K,p,nbrOfRealizations)
%%=============================================================
%The file is used to design DL-OBE combining vectors based on Monte-Carlo method-based OBE matrices in Corollary 3 of the paper:
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



V_OBE_Combining_Distributed_Local_Monte = zeros(M*N,nbrOfRealizations,K);
W_OBE_matrix = zeros(N,N,M,K);

Term1 = zeros((N)^2,(N)^2,M,K);
Term2 = zeros(N,N,M,K);
Term3 = zeros(N,N,M,K);
Term4 = zeros((N)^2,(N)^2,M,K);


%% Centralized OBE combining design based on Monte-carlo

for n = 1:nbrOfRealizations

    Gallj = reshape(G(:,n,:),[M*N K]);
    Ghatallj = reshape(Ghat_MMSE(:,n,:),[M*N K]);


    for k = 1:K

        for m = 1:M

            ghat_mk = Ghatallj((m-1)*N+1:m*N,k);
            g_mk = Gallj((m-1)*N+1:m*N,k);


            Term3(:,:,m,k) = Term3(:,:,m,k) + ghat_mk*ghat_mk'/nbrOfRealizations;

            Term2(:,:,m,k) = Term2(:,:,m,k) + g_mk*ghat_mk'/nbrOfRealizations;


            for l = 1:K

                g_ml = Gallj((m-1)*N+1:m*N,l);

                Term1(:,:,m,k) = Term1(:,:,m,k) + p(l)*kron((ghat_mk*ghat_mk').',(g_ml*g_ml'))/nbrOfRealizations;

            end

        end
    end
end


% Term1 = sum(Term1,4);

for k = 1:K
    for m = 1:M

    Term2_mk = Term2(:,:,m,k);
    Term2_mk = Term2_mk(:);

    Term3_mk = Term3(:,:,k);
    Term4(:,:,m,k) = Term2_mk*Term2_mk';


    W_OBE_k_vector = (Term1(:,:,m,k) - p(k)*(Term2_mk*Term2_mk') + kron(Term3_mk.',eye(N)))\Term2_mk;


    W_OBE_matrix(:,:,m,k) = reshape(W_OBE_k_vector,[N,N]);

    V_OBE_Combining_Distributed_Local_Monte((m-1)*N+1:m*N,:,k) = W_OBE_matrix(:,:,m,k)*Ghat_MMSE((m-1)*N+1:m*N,:,k);

    clear W_OBE_k_vector Term2_k Term3_k

    end

end


