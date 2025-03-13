%%=============================================================
%The file is used to generates the data applied in Fig. 4 of the paper:
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

clc
clear all
close all

tic

M = 20;
N = 2;
K_total = 10:5:30;
nbrOfSetups = 100;

nbrOfRealizations = 1000;

tau_p_total = [1,5];
tau_c = 200;

%Uplink transmit power per UE (W)
p = 0.2; %200 mW
%Create the power vector for all UEs (The uplink power is the same
%(p)at each UE)



SE_Centralized_MMSE_Sum = zeros(nbrOfSetups,length(K_total),length(tau_p_total));
SE_Centralized_OBE_Monte_Sum = zeros(nbrOfSetups,length(K_total),length(tau_p_total));


% % % % % % % ----Parallel computing
core_number = 2;           
parpool('local',core_number);
% % % Starting parallel pool (parpool) using the 'local' profile ...

for n = 1:length(tau_p_total)

    tau_p = tau_p_total(n);


    for k = 1:length(K_total)

        K = K_total(k);
        pv = p*ones(1,K);

        parfor i = 1:nbrOfSetups


            [R_AP,H_LoS_Single_real,channelGain,channelGain_LoS,channelGain_NLoS] = functionGenerateSetupDeploy(M,K,N,1,1);
            [H,H_LoS] = functionChannelGeneration(R_AP,H_LoS_Single_real,M,K,N,nbrOfRealizations);

            A_singleLayer = reshape(repmat(eye(M),1,K),M,M,K);

            [Pset] = functionPilotAllocation(R_AP,H_LoS_Single_real,A_singleLayer,M,K,N,tau_p,pv);


            [Hhat_MMSE] = functionChannelEstimates_MMSE(R_AP,H_LoS,H,nbrOfRealizations,M,K,N,tau_p,pv,Pset);


            [Rhat,Phi,C_MMSE,C_total,C_total_blk] = functionMatrixGeneration(R_AP,pv,M,K,N,tau_p,Pset);


            [V_MMSE_Combining] = functionMMSE_Combining_Centralized(Hhat_MMSE,C_total_blk,nbrOfRealizations,M,N,K,pv);
            [SE_Centralized_MMSE_Monte_i] = functionComputeSE_Centralized(Hhat_MMSE,V_MMSE_Combining,C_total_blk,tau_c,tau_p,nbrOfRealizations,M,N,K,pv);


%             [V_OBE_Combining_Centralized,W_OBE_matrix_Centralized] = functionOBE_Combining_Centralized_Analytical(H_LoS_Single_real,Hhat_MMSE,Rhat,Phi,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);
%             [SE_Centralized_OBE_Monte_i] = functionComputeSE_Centralized(Hhat_MMSE,V_OBE_Combining_Centralized,C_total_blk,tau_c,tau_p,nbrOfRealizations,M,N,K,pv);

            %--DG-OBE can achieve similar SE with the C-OBE

            [V_OBE_Combining_Distributed,~] = functionOBE_Combining_Distributed(H_LoS_Single_real,Hhat_MMSE,Rhat,Phi,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);
            [SE_Centralized_OBE_Monte_i] = functionComputeSE_Centralized(Hhat_MMSE,V_OBE_Combining_Distributed,C_total_blk,tau_c,tau_p,nbrOfRealizations,M,N,K,pv);




            SE_Centralized_MMSE_Sum(i,k,n) = sum(SE_Centralized_MMSE_Monte_i);
            SE_Centralized_OBE_Monte_Sum(i,k,n) = sum(SE_Centralized_OBE_Monte_i);

            disp([num2str(i) ' setups out of ' num2str(nbrOfSetups)]);

        end



    end
    disp([num2str(k) ' -th number of K out of ' num2str(K_total)]); 
end







toc




 