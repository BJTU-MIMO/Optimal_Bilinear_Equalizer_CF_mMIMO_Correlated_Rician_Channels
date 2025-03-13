function [V_OBE_Combining_Distributed_Monte] = functionOBE_Combining_Distributed_Monte(G,Ghat_MMSE,M,N,K,p,nbrOfRealizations)
%%=============================================================
%The file is used to design DG-OBE combining vectors based on Monte-Carlo method-based OBE matrices in Corollary 2 of the paper:
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



% V_OBE_Combining_Centralized_Monte = zeros(M*N,nbrOfRealizations,K);
W_OBE_matrix = zeros(N,N,M,K);



zkl = zeros(M*N^2,K,K);

Theta5 = zeros(M*N^2,M*N^2,K);
Term1 = zeros(M*N^2,M*N^2,K);
Term2 = zeros(M*N^2,K);
Term3 = zeros(N,N,M,K);

V_OBE_Combining_Distributed_Monte = zeros(M*N,nbrOfRealizations,K);

%% Centralized OBE combining design based on Monte-carlo

for n = 1:nbrOfRealizations

    Gallj = reshape(G(:,n,:),[M*N K]);
    Ghatallj = reshape(Ghat_MMSE(:,n,:),[M*N K]);


    for k = 1:K

        ghat_k = Ghatallj(:,k);
        g_k = Gallj(:,k);



        for m = 1:M

            hhm = g_k((m-1)*N+1:m*N)*ghat_k((m-1)*N+1:m*N)';


            Term3(:,:,m,k) = Term3(:,:,m,k) + ghat_k((m-1)*N+1:m*N)*ghat_k((m-1)*N+1:m*N)'/nbrOfRealizations;


            Term2((m-1)*N^2+1:m*N^2,k) = Term2((m-1)*N^2+1:m*N^2,k) + hhm(:)/nbrOfRealizations;

        end



        for l = 1:K

            for m = 1:M

                g_l = Gallj(:,l);

                zzm = g_l((m-1)*N+1:m*N)*ghat_k((m-1)*N+1:m*N)';

                zkl((m-1)*N^2+1:m*N^2,k,l) = zzm(:);

            end

            Term1(:,:,k) = Term1(:,:,k) + p(l)*(zkl(:,k,l)*zkl(:,k,l)')/nbrOfRealizations;



        end
    end
end



for m = 1:M
    for k = 1:K

        Theta5((m-1)*N^2+1:m*N^2,(m-1)*N^2+1:m*N^2,k) = kron(Term3(:,:,m,k).',eye(N));

    end
end
     

for k = 1:K

    Term1_k = Term1(:,:,k);
    Term2_k = p(k)*Term2(:,k)*Term2(:,k)';
    Theta5_k = Theta5(:,:,k);

    W_OBE_k_vector = (Term1_k- Term2_k + Theta5_k)\Term2(:,k);

     for m = 1:M
         
         W_OBE_matrix(:,:,m,k) = reshape(W_OBE_k_vector((m-1)*N^2+1:m*N^2),[N,N]);

         V_OBE_Combining_Distributed_Monte((m-1)*N+1:m*N,:,k) = W_OBE_matrix(:,:,m,k)*Ghat_MMSE((m-1)*N+1:m*N,:,k);

     end
   

    clear W_OBE_k_vector Term2_k Term3_k

end


