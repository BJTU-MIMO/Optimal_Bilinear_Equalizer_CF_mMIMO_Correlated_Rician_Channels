function [coherentx,nonCoherentx] = functionMMSE_interferenceLevels( R_AP,HMean_Withoutphase,A,M,K,N,tau_p,p,Pset)
%%=============================================================
%The file is used to check the levels of coherent and non-coherent interference levels (used
%for pilot allocation) of the paper:
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



%Prepare to store the results
Ksi = zeros(M,M,K,K); %Ksi
X_p1 = zeros(M,M,K,K); %X(1)
nonCoherent = zeros(K,K);
coherent = zeros(K,K);



%Generate matrix used in this setup

Lk = zeros(N,N,M,K);
Phi = zeros(N,N,M,K);
Omega = zeros(N,N,M,K);

for m = 1:M
    for k = 1:K
       
         Lk(:,:,m,k) = HMean_Withoutphase((m-1)*N+1:m*N,k)*(HMean_Withoutphase((m-1)*N+1:m*N,k))';
         
    end
end

%Go through all APs          
for m = 1:M
    
    %Go through all UEs
    for k = 1:K
        
        %Compute the UEs indexes that use the same pilot as UE k
        inds = Pset(:,k);
        PsiInv = zeros(N,N);
        
        %Go through all UEs that use the same pilot as UE k 
        for z = 1:length(inds)   
            
            PsiInv = PsiInv + p(inds(z))*tau_p*R_AP(:,:,m,inds(z));
        
        end
            PsiInv = PsiInv + eye(N);
            
            for z = 1:length(inds)
                
                Phi(:,:,m,inds(z)) = PsiInv;
            
            end
            
            Omega(:,:,m,k) = R_AP(:,:,m,k)/PsiInv*R_AP(:,:,m,k);
            
    end
end



% Go through all APs
for m = 1:M
    
    %Go through all UEs
    for k = 1:K
        
        for l=1:K  %Non-coherent interference (i=k')
            
            Ksi(m,m,k,l) = p(k)*tau_p*trace(R_AP(:,:,m,l)*Omega(:,:,m,k))+...
                trace(Lk(:,:,m,k)*R_AP(:,:,m,l)) + p(k)*tau_p*trace(Lk(:,:,m,l)*Omega(:,:,m,k))+...
                trace(Lk(:,:,m,k))*trace(Lk(:,:,m,k));
            
           
           if any(l==Pset(:,k)) %Coherent interference (If there is pilot contamination)
           
           X_p1(m,m,k,l) = sqrt(p(k)*p(l))*tau_p*trace(R_AP(:,:,m,l)/Phi(:,:,m,k)*R_AP(:,:,m,k));
           

           end
           
        end
        
    end
    
end

%Go through all UEs
for k = 1:K
    
    for l=1:K  %Non-coherent interference (i=k')
    
        nonCoherent(k,l) = p(l)*trace(A(:,:,k)'*Ksi(:,:,k,l)*A(:,:,k));
        
        if any(l==Pset(:,k)) 
            
            coherent(k,l)=  p(l)*abs(trace(A(:,:,k)*X_p1(:,:,k,l)))^2; 
            
        end
        
    end
    
end

coherentx = sum(coherent,2);
nonCoherentx = sum(nonCoherent,2);
              
end
    
