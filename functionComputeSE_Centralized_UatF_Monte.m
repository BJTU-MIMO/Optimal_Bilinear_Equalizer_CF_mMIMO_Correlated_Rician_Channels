function [SE_Centralized_UatF,denominator_termkl,numerator_k,denominator_term2,denominator_1] = functionComputeSE_Centralized_UatF_Monte(H,V_Combining,tau_c,tau_p,nbrOfRealizations,M,N,K,p)
%%=============================================================
%The file is used to compute the achievable SE for the centralized processing by Monte-Carlo methods based on
%the UatF capacity bound of the paper:
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
SE_Centralized_UatF = zeros(K,1);



numerator_term = zeros(K,1);
denominator_termkl = zeros(K,K);
denominator_term2 = zeros(K,1);
numerator_k = zeros(K,1);

%% Go through all channel realizations
for n = 1:nbrOfRealizations
    
    
    
    %Extract channel estimate and combining vector realizations from all UEs to all APs
    Hallj = reshape(H(:,n,:),[M*N K]);
   
    V_n = reshape(V_Combining(:,n,:),[M*N K]);
    

    %Go through all UEs

    for k = 1:K

        
        v = V_n(:,k); %Extract combining vector
        
        %Compute the numerator term E{v^H*gk}
        numerator_term(k) = numerator_term(k) + (v'*Hallj(:,k))/nbrOfRealizations;

        %Compute the denominator term E{|v|^2}
        denominator_term2(k) = denominator_term2(k) + norm(v).^2/nbrOfRealizations;

        for l = 1:K
            
            %Compute the denominator term E{|v^H*gl|^2} 
            denominator_termkl(k,l) = denominator_termkl(k,l) + p(l)*abs(v'*Hallj(:,l))^2/nbrOfRealizations;

        end

    end

        
end
    


for k = 1:K

    numerator_k(k) = p(k)*numerator_term(k)*numerator_term(k)';


    denominator_1 = sum(denominator_termkl,2);
    denominator_1_k = denominator_1(k);

    denominator_term2_k = denominator_term2(k);

    denominator_k = denominator_1_k - numerator_k(k) + denominator_term2_k;

    SE_Centralized_UatF(k) = prelogFactor*real(log2(1+ numerator_k(k)/denominator_k));

end







