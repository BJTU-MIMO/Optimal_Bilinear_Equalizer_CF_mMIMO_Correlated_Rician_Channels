function [V_MMSE_Combining_Asymptotic] = functionMMSE_Combining_Asymptotic(Hhat,G_LoS_eff,R,Phi,Pset,C_tot,nbrOfRealizations,M,N,K,tau_p,p)

% The file designs centralized MMSE combining vectors 



%Store identity matrices of different sizes
eyeMN = eye(M*N);
eyeK = eye(K);


%Diagonal matrix with transmit powers and its square root
Dp = diag(p);

G_LoS_eff = reshape(G_LoS_eff(:,1,:),M*N,K);

V_MMSE_Combining_Asymptotic = zeros(M*N,nbrOfRealizations,K);


Z = C_tot + eyeMN;

Term1 = zeros(K,K);

R_blk = zeros(M*N,M*N,K);
Phi_blk = zeros(M*N,M*N,K);

for k = 1:K

    for m = 1:M

        R_blk((m-1)*N+1:m*N,(m-1)*N+1:m*N,k) = R(:,:,m,k);
        Phi_blk((m-1)*N+1:m*N,(m-1)*N+1:m*N,k) = Phi(:,:,m,k);

    end
end



for k = 1:K

    for l = 1:K

        if any(l == Pset(:,k))

            Term1(k,l) = 1/(M*N)*G_LoS_eff(:,k)'*inv(Z)*G_LoS_eff(:,l) + 1/(M*N)*sqrt(p(k)*p(l))*tau_p*trace(R_blk(:,:,l)/Phi_blk(:,:,k)*R_blk(:,:,k)*inv(Z));

        else

            Term1(k,l) = 1/(M*N)*G_LoS_eff(:,k)'*inv(Z)*G_LoS_eff(:,l);


        end
    end
end



for k = 1:K

    for m = 1:M

        for n = 1:nbrOfRealizations

            Hhat_m = reshape(Hhat((m-1)*N+1:m*N,n,:),[N K]);
            V_MMSE_Combining_Asymptotic((m-1)*N+1:m*N,n,k) = p(k)*inv(Z((m-1)*N+1:m*N,(m-1)*N+1:m*N))*Hhat_m*inv(Term1 + 1/(M*N)*inv(Dp))*eyeK(:,k);

        end
    end
end







