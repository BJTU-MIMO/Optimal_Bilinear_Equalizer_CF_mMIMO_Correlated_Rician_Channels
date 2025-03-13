function [SE,V_OBE_Combining_Centralized_Monte,W_OBE_matrix,Term1,Term4,Term3] = functionOBE_Combining_Centralized_Monte(G,Ghat_MMSE,M,N,K,tau_c,tau_p,p,nbrOfRealizations)
%%=============================================================
%The file is used to design centralized OBE combining vectors based on Monte-Carlo method-based OBE matrices in Corollary 1 of the paper:
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


V_OBE_Combining_Centralized_Monte = zeros(M*N,nbrOfRealizations,K);
W_OBE_matrix = zeros(M*N,M*N,K);

Term1 = zeros((M*N)^2,(M*N)^2,K);
Term2 = zeros((M*N),(M*N),K);
Term3 = zeros((M*N),(M*N),K);
Term4 = zeros((M*N)^2,(M*N)^2,K);
SE = zeros(K,1);
prelogFactor = (1-tau_p/tau_c);
%% Centralized OBE combining design based on Monte-carlo

for n = 1:nbrOfRealizations

    Gallj = reshape(G(:,n,:),[M*N K]);
    Ghatallj = reshape(Ghat_MMSE(:,n,:),[M*N K]);


    for k = 1:K

        ghat_k = Ghatallj(:,k);
        g_k = Gallj(:,k);


        Term2(:,:,k) = Term2(:,:,k) + g_k*ghat_k'/nbrOfRealizations;

        Term3(:,:,k) = Term3(:,:,k) + ghat_k*ghat_k'/nbrOfRealizations;


        for l = 1:K

            g_l = Gallj(:,l);

            Term1(:,:,k) = Term1(:,:,k) + p(l)*kron((ghat_k*ghat_k').',(g_l*g_l'))/nbrOfRealizations;

        end
    end
end


% Term1 = sum(Term1,4);

for k = 1:K

    Term2_k = Term2(:,:,k);
    Term2_k = Term2_k(:);

    Term3_k = Term3(:,:,k);
%     Term3_k = Term3_k(:);
Term4(:,:,k) = Term2_k*Term2_k';


    W_OBE_k_vector = (Term1(:,:,k) - p(k)*(Term2_k*Term2_k') + kron(Term3_k.',eye(M*N)))\Term2_k;


    W_OBE_matrix(:,:,k) = reshape(W_OBE_k_vector,[M*N,M*N]);

    V_OBE_Combining_Centralized_Monte(:,:,k) = W_OBE_matrix(:,:,k)*Ghat_MMSE(:,:,k);

    SE(k) = prelogFactor*real(log2(1+ p(k)*Term2_k'*W_OBE_k_vector));
    clear W_OBE_k_vector Term2_k Term3_k

end


