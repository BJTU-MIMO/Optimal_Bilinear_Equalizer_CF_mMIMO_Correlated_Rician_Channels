function [SE_Distributed] = functionComputeSE_Distributed_Standard(Hhat,V_Combining,C_total,tau_c,tau_p,nbrOfRealizations,M,N,K,p)



% The file computes SE performance for the centralized processing based on
% the standard capacity lower bound


%If only one transmit power is provided, use the same for all the UEs
if length(p) == 1
   p = p*ones(K,1);
end


%Compute the prelog factor
prelogFactor = (1-tau_p/tau_c);

%Prepare to store simulation results
SE_Distributed = zeros(K,M);

eyeN = eye(N);


%Diagonal matrix with transmit powers and its square root
Dp12 = diag(sqrt(p));



%% Go through all channel realizations
for m = 1:M
    for n = 1:nbrOfRealizations



        %Extract channel estimate and combining vector realizations from all UEs to all APs
        Hhatallj = reshape(Hhat(1+(m-1)*N:m*N,n,:),[N K]);

        V_n = reshape(V_Combining(1+(m-1)*N:m*N,n,:),[N K]);


        %Go through all UEs
        for k = 1:K

            v = V_n(:,k); %Extract combining vector

            %Compute numerator and denominator of instantaneous SINR at Level 4
            numerator = p(k)*abs(v'*Hhatallj(:,k))^2;
            denominator = norm(v'*Hhatallj*Dp12)^2 + v'*(C_total(:,:,m)+eyeN)*v - numerator;

            %Compute instantaneous SE for one channel realization
            SE_Distributed(k,m) = SE_Distributed(k,m)+ prelogFactor*real(log2(1+numerator/denominator))/nbrOfRealizations;

        end



    end
    
  
end



