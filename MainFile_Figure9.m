%%=============================================================
%The file is used to generates the data applied in Fig. 9 of the paper:
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
N = 4;
K = 10;


nbrOfRealizations = 1000;
nbrOfSetups = 30;

tau_p = 1;
tau_c = 200;


%Uplink transmit power per UE (W)
p = 0.2; %200 mW
%Create the power vector for all UEs (The uplink power is the same
%(p)at each UE)
pv = p*ones(1,K);




%--MMSE
SE_LSFD_LMMSE_MMSE_total = zeros(K,nbrOfSetups);
SE_LMMSE_MMSE_total = zeros(K,nbrOfSetups);
SE_DG_OBE_LSFD_Monte_MMSE_total = zeros(K,nbrOfSetups);
SE_DG_OBE_Monte_MMSE_total = zeros(K,nbrOfSetups);
SE_DL_OBE_LSFD_Monte_MMSE_total = zeros(K,nbrOfSetups);
SE_DL_OBE_Monte_local_MMSE_total = zeros(K,nbrOfSetups);
SE_LSFD_MR_MMSE_total = zeros(K,nbrOfSetups);
SE_MR_Distributed_MMSE_total = zeros(K,nbrOfSetups);

%--LMMSE
SE_LSFD_LMMSE_LMMSE_total = zeros(K,nbrOfSetups);
SE_LMMSE_LMMSE_total = zeros(K,nbrOfSetups);
SE_DG_OBE_LSFD_Monte_LMMSE_total = zeros(K,nbrOfSetups);
SE_DG_OBE_Monte_LMMSE_total = zeros(K,nbrOfSetups);
SE_DL_OBE_LSFD_Monte_LMMSE_total = zeros(K,nbrOfSetups);
SE_DL_OBE_Monte_local_LMMSE_total = zeros(K,nbrOfSetups);
SE_LSFD_MR_LMMSE_total = zeros(K,nbrOfSetups);
SE_MR_Distributed_LMMSE_total = zeros(K,nbrOfSetups);


%--LS
SE_LSFD_LMMSE_LS_total = zeros(K,nbrOfSetups);
SE_LMMSE_LS_total = zeros(K,nbrOfSetups);
SE_DG_OBE_LSFD_Monte_LS_total = zeros(K,nbrOfSetups);
SE_DG_OBE_Monte_LS_total = zeros(K,nbrOfSetups);
SE_DL_OBE_LSFD_Monte_LS_total = zeros(K,nbrOfSetups);
SE_DL_OBE_Monte_local_LS_total = zeros(K,nbrOfSetups);
SE_LSFD_MR_LS_total = zeros(K,nbrOfSetups);
SE_MR_Distributed_LS_total = zeros(K,nbrOfSetups);

% % % % % % % ----Parallel computing
core_number = 2;           
parpool('local',core_number);
% % % Starting parallel pool (parpool) using the 'local' profile ...

parfor i = 1:nbrOfSetups

    [R_AP,H_LoS_Single_real,channelGain,channelGain_LoS,channelGain_NLoS] = functionGenerateSetupDeploy(M,K,N,1,1);

    [H,H_LoS,PhaseMatrix] = functionChannelGeneration(R_AP,H_LoS_Single_real,M,K,N,nbrOfRealizations);

    A_singleLayer = reshape(repmat(eye(M),1,K),M,M,K);
    [Pset] = functionPilotAllocation(R_AP,H_LoS_Single_real,A_singleLayer,M,K,N,tau_p,pv);


    %--MMSE Estimator
    [Hhat_MMSE] = functionChannelEstimates_MMSE(R_AP,H_LoS,H,nbrOfRealizations,M,K,N,tau_p,pv,Pset);

    [Rhat,Phi,C_MMSE,C_total,C_total_blk] = functionMatrixGeneration(R_AP,pv,M,K,N,tau_p,Pset);


    [V_MMSE_Combining_Distributed] = functionMMSE_Combining_Distributed(Hhat_MMSE,C_total,nbrOfRealizations,M,N,K,pv);
    [SE_LSFD_LMMSE_MMSE,SE_LMMSE_MMSE] = functionComputeSE_Distributed_Monte(H,V_MMSE_Combining_Distributed,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);


    [V_DG_OBE_Combining_Distributed_Monte_MMSE] = functionOBE_Combining_Distributed_Monte(H,Hhat_MMSE,M,N,K,pv,nbrOfRealizations);
    [SE_DG_OBE_LSFD_Monte_MMSE,SE_DG_OBE_Monte_MMSE] = functionComputeSE_Distributed_Monte(H,V_DG_OBE_Combining_Distributed_Monte_MMSE,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);


    [V_OBE_Combining_Distributed_Local_Monte_MMSE] = functionOBE_Combining_Distributed_Local_Monte(H,Hhat_MMSE,M,N,K,pv,nbrOfRealizations);
    [SE_DL_OBE_LSFD_Monte_MMSE,SE_DL_OBE_Monte_local_MMSE] = functionComputeSE_Distributed_Monte(H,V_OBE_Combining_Distributed_Local_Monte_MMSE,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);


    [SE_LSFD_MR_MMSE,SE_MR_Distributed_MMSE] = functionComputeSE_Distributed_Monte(H,Hhat_MMSE,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);


    SE_LSFD_LMMSE_MMSE_total(:,i) = SE_LSFD_LMMSE_MMSE;
    SE_LMMSE_MMSE_total(:,i) = SE_LMMSE_MMSE;
    SE_DG_OBE_LSFD_Monte_MMSE_total(:,i) = SE_DG_OBE_LSFD_Monte_MMSE;
    SE_DG_OBE_Monte_MMSE_total(:,i) = SE_DG_OBE_Monte_MMSE;
    SE_DL_OBE_LSFD_Monte_MMSE_total(:,i) = SE_DL_OBE_LSFD_Monte_MMSE;
    SE_DL_OBE_Monte_local_MMSE_total(:,i) = SE_DL_OBE_Monte_local_MMSE;
    SE_LSFD_MR_MMSE_total(:,i) = SE_LSFD_MR_MMSE;
    SE_MR_Distributed_MMSE_total(:,i) = SE_MR_Distributed_MMSE;


    %--LMMSE Estimator

    [Hhat_LMMSE] = functionChannelEstimates_LMMSE(R_AP,H_LoS_Single_real,H,nbrOfRealizations,M,K,N,tau_p,pv,Pset);
    [C_LMMSE,C_LMMSE_total,C_LMMSE_total_blk] = functionMatrixGeneration_LMMSE(R_AP,H_LoS_Single_real,pv,M,K,N,tau_p,Pset);

    [V_MMSE_Combining_Distributed_LMMSE] = functionMMSE_Combining_Distributed(Hhat_LMMSE,C_LMMSE_total,nbrOfRealizations,M,N,K,pv);
    [SE_LSFD_LMMSE_LMMSE,SE_LMMSE_LMMSE] = functionComputeSE_Distributed_Monte(H,V_MMSE_Combining_Distributed_LMMSE,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);


    [V_DG_OBE_Combining_Distributed_Monte_LMMSE] = functionOBE_Combining_Distributed_Monte(H,Hhat_LMMSE,M,N,K,pv,nbrOfRealizations);
    [SE_DG_OBE_LSFD_Monte_LMMSE,SE_DG_OBE_Monte_LMMSE] = functionComputeSE_Distributed_Monte(H,V_DG_OBE_Combining_Distributed_Monte_LMMSE,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);


    [V_OBE_Combining_Distributed_Local_Monte_LMMSE] = functionOBE_Combining_Distributed_Local_Monte(H,Hhat_LMMSE,M,N,K,pv,nbrOfRealizations);
    [SE_DL_OBE_LSFD_Monte_LMMSE,SE_DL_OBE_Monte_local_LMMSE] = functionComputeSE_Distributed_Monte(H,V_OBE_Combining_Distributed_Local_Monte_LMMSE,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);


    [SE_LSFD_MR_LMMSE,SE_MR_Distributed_LMMSE] = functionComputeSE_Distributed_Monte(H,Hhat_LMMSE,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);

    SE_LSFD_LMMSE_LMMSE_total(:,i) = SE_LSFD_LMMSE_LMMSE;
    SE_LMMSE_LMMSE_total(:,i) = SE_LMMSE_LMMSE;
    SE_DG_OBE_LSFD_Monte_LMMSE_total(:,i) = SE_DG_OBE_LSFD_Monte_LMMSE;
    SE_DG_OBE_Monte_LMMSE_total(:,i) = SE_DG_OBE_Monte_LMMSE;
    SE_DL_OBE_LSFD_Monte_LMMSE_total(:,i) = SE_DL_OBE_LSFD_Monte_LMMSE;
    SE_DL_OBE_Monte_local_LMMSE_total(:,i) = SE_DL_OBE_Monte_local_LMMSE;
    SE_LSFD_MR_LMMSE_total(:,i) = SE_LSFD_MR_LMMSE;
    SE_MR_Distributed_LMMSE_total(:,i) = SE_MR_Distributed_LMMSE;

    %--LS Estimator
    [Hhat_LS] = functionChannelEstimates_LS(H,nbrOfRealizations,M,K,N,tau_p,pv,Pset);
    [C_LS,C_LS_total,C_LS_total_blk] = functionMatrixGeneration_LS(R_AP,H_LoS_Single_real,pv,M,K,N,tau_p,Pset);


    [V_MMSE_Combining_Distributed_LS] = functionMMSE_Combining_Distributed(Hhat_LS,C_LS_total,nbrOfRealizations,M,N,K,pv);
    [SE_LSFD_LMMSE_LS,SE_LMMSE_LS] = functionComputeSE_Distributed_Monte(H,V_MMSE_Combining_Distributed_LS,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);


    [V_DG_OBE_Combining_Distributed_Monte_LS] = functionOBE_Combining_Distributed_Monte(H,Hhat_LS,M,N,K,pv,nbrOfRealizations);
    [SE_DG_OBE_LSFD_Monte_LS,SE_DG_OBE_Monte_LS] = functionComputeSE_Distributed_Monte(H,V_DG_OBE_Combining_Distributed_Monte_LS,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);


    [V_OBE_Combining_Distributed_Local_Monte_LS] = functionOBE_Combining_Distributed_Local_Monte(H,Hhat_LS,M,N,K,pv,nbrOfRealizations);
    [SE_DL_OBE_LSFD_Monte_LS,SE_DL_OBE_Monte_local_LS] = functionComputeSE_Distributed_Monte(H,V_OBE_Combining_Distributed_Local_Monte_LS,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);

    [SE_LSFD_MR_LS,SE_MR_Distributed_LS] = functionComputeSE_Distributed_Monte(H,Hhat_LS,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);

    SE_LSFD_LMMSE_LS_total(:,i) = SE_LSFD_LMMSE_LS;
    SE_LMMSE_LS_total(:,i) = SE_LMMSE_LS;
    SE_DG_OBE_LSFD_Monte_LS_total(:,i) = SE_DG_OBE_LSFD_Monte_LS;
    SE_DG_OBE_Monte_LS_total(:,i) = SE_DG_OBE_Monte_LS;
    SE_DL_OBE_LSFD_Monte_LS_total(:,i) = SE_DL_OBE_LSFD_Monte_LS;
    SE_DL_OBE_Monte_local_LS_total(:,i) = SE_DL_OBE_Monte_local_LS;
    SE_LSFD_MR_LS_total(:,i) = SE_LSFD_MR_LS;
    SE_MR_Distributed_LS_total(:,i) = SE_MR_Distributed_LS;

    disp([num2str(i) ' setups out of ' num2str(nbrOfSetups)]); 



end







toc

