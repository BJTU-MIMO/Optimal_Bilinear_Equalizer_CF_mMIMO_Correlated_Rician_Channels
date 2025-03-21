%%=============================================================
%The file is used to generates the data applied in Fig. 1 of the paper:
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
N = 1:1:6;
K = 20;
nbrOfSetups = 30;
nbrOfRealizations = 1000;

tau_p = 1;
tau_c = 200;

%Uplink transmit power per UE (W)
p = 0.2; %200 mW
%Create the power vector for all UEs (The uplink power is the same
%(p)at each UE)
pv = p*ones(1,K);


SE_Centralized_UatF_MMSE = zeros(K,nbrOfSetups,length(N));
SE_Centralized_UatF_OBE_Monte = zeros(K,nbrOfSetups,length(N));
SE_Centralized_UatF_MR_Monte = zeros(K,nbrOfSetups,length(N));
SE_Centralized_UatF_OBE_Analytical = zeros(K,nbrOfSetups,length(N));
SE_Centralized_UatF_MR_Analytical = zeros(K,nbrOfSetups,length(N));

% % % % % % % ----Parallel computing
core_number = 2;           
parpool('local',core_number);
% % % Starting parallel pool (parpool) using the 'local' profile ...

for n = 1:length(N)
    parfor i = 1:nbrOfSetups
        
        [R_AP,H_LoS_Single_real,channelGain,channelGain_LoS,channelGain_NLoS] = functionGenerateSetupDeploy(M,K,N(n),1,1);
        [H,H_LoS] = functionChannelGeneration(R_AP,H_LoS_Single_real,M,K,N(n),nbrOfRealizations);
        
        A_singleLayer = reshape(repmat(eye(M),1,K),M,M,K);
        
        [Pset] = functionPilotAllocation(R_AP,H_LoS_Single_real,A_singleLayer,M,K,N(n),tau_p,pv);


        [Hhat_MMSE] = functionChannelEstimates_MMSE(R_AP,H_LoS,H,nbrOfRealizations,M,K,N(n),tau_p,pv,Pset);


        [Rhat,Phi,C_MMSE,C_total,C_total_blk] = functionMatrixGeneration(R_AP,pv,M,K,N(n),tau_p,Pset);

        %--C-MMSE
        [V_MMSE_Combining] = functionMMSE_Combining_Centralized(Hhat_MMSE,C_total_blk,nbrOfRealizations,M,N(n),K,pv);
        [SE_Centralized_UatF_MMSE_i] = functionComputeSE_Centralized_UatF_Monte(H,V_MMSE_Combining,tau_c,tau_p,nbrOfRealizations,M,N(n),K,pv);
        

        %--C-OBE

        [V_OBE_Combining_Centralized,W_OBE_matrix_Centralized] = functionOBE_Combining_Centralized_Analytical(H_LoS_Single_real,Hhat_MMSE,Rhat,Phi,R_AP,Pset,M,N(n),K,pv,tau_p,nbrOfRealizations);
        [SE_Centralized_UatF_OBE_Monte_i] = functionComputeSE_Centralized_UatF_Monte(H,V_OBE_Combining_Centralized,tau_c,tau_p,nbrOfRealizations,M,N(n),K,pv);
        [SE_Centralized_UatF_OBE_Analytical_i] = functionComputeSE_Centralized_UatF_Analytical(W_OBE_matrix_Centralized,H_LoS_Single_real,Rhat,R_AP,Phi,Pset,tau_c,tau_p,M,N,K,pv);


        %--C-MR
        W_MR = reshape(repmat(eye(M*N(n)),1,K),M*N(n),M*N(n),K);
        [SE_Centralized_UatF_MR_Monte_i] = functionComputeSE_Centralized_UatF_Monte(H,Hhat_MMSE,tau_c,tau_p,nbrOfRealizations,M,N(n),K,pv);
        [SE_Centralized_UatF_MR_Analytical_i] = functionComputeSE_Centralized_UatF_Analytical(W_MR,H_LoS_Single_real,Rhat,R_AP,Phi,Pset,tau_c,tau_p,M,N(n),K,pv);



        SE_Centralized_UatF_MMSE(:,i,n) = SE_Centralized_UatF_MMSE_i;
        SE_Centralized_UatF_OBE_Monte(:,i,n) = SE_Centralized_UatF_OBE_Monte_i;
        SE_Centralized_UatF_MR_Monte(:,i,n) = SE_Centralized_UatF_MR_Monte_i;

        SE_Centralized_UatF_OBE_Analytical(:,i,n) = SE_Centralized_UatF_OBE_Analytical_i;
        SE_Centralized_UatF_MR_Analytical(:,i,n) = SE_Centralized_UatF_MR_Analytical_i;



        disp([num2str(i) ' setups out of ' num2str(nbrOfSetups)]); 



    end
    disp([num2str(n) ' -th number of antennas out of ' num2str(N)]); 
end





toc




 