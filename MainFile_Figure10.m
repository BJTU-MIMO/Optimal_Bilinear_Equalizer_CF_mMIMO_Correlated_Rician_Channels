%%=============================================================
%The file is used to generates the data applied in Fig. 10 of the paper:
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
K = 20;


nbrOfRealizations = 1000;
nbrOfSetups = 30;

tau_p = 1;
tau_c = 200;

%Uplink transmit power per UE (W)
p = 0.2; %200 mW
%Create the power vector for all UEs (The uplink power is the same
%(p)at each UE)
pv = p*ones(1,K);




SE_MMSE_Centralized_Standard_with_PS_total = zeros(K,nbrOfSetups);
SE_MMSE_Centralized_UatF_with_PS_total = zeros(K,nbrOfSetups);
SE_OBE_Centralized_UatF_Monte_with_PS_total = zeros(K,nbrOfSetups);
SE_OBE_Centralized_OBE_Standard_Monte_with_PS_total = zeros(K,nbrOfSetups);
SE_OBE_Centralized_UatF_Analytical_with_PS_total = zeros(K,nbrOfSetups);

SE_MMSE_Centralized_Standard_wo_PS_total = zeros(K,nbrOfSetups);
SE_MMSE_Centralized_UatF_wo_PS_total = zeros(K,nbrOfSetups);
SE_OBE_Centralized_UatF_Monte_wo_PS_total = zeros(K,nbrOfSetups);
SE_OBE_Centralized_OBE_Standard_Monte_wo_PS_total = zeros(K,nbrOfSetups);
SE_OBE_Centralized_UatF_Analytical_wo_PS_total = zeros(K,nbrOfSetups);


SE_LMMSE_LSFD_Monte_with_PS_total = zeros(K,nbrOfSetups);

SE_OBE_DG_LSFD_Monte_with_PS_total = zeros(K,nbrOfSetups);
SE_OBE_DG_Distribued_Monte_with_PS_total = zeros(K,nbrOfSetups);
SE_OBE_DG_LSFD_Analytical_with_PS_total = zeros(K,nbrOfSetups);
SE_OBE_DG_Distribued_Analytical_with_PS_total = zeros(K,nbrOfSetups);

SE_OBE_DL_LSFD_Monte_with_PS_total = zeros(K,nbrOfSetups);
SE_OBE_DL_Distribued_Monte_with_PS_total = zeros(K,nbrOfSetups);
SE_OBE_DL_LSFD_Analytical_with_PS_total = zeros(K,nbrOfSetups);
SE_OBE_DL_Distribued_Analytical_with_PS_total = zeros(K,nbrOfSetups);

SE_LMMSE_LSFD_Monte_wo_PS_total = zeros(K,nbrOfSetups);

SE_OBE_DG_LSFD_Monte_wo_PS_total = zeros(K,nbrOfSetups);
SE_OBE_DG_Distribued_Monte_wo_PS_total = zeros(K,nbrOfSetups);
SE_OBE_DG_LSFD_Analytical_wo_PS_total = zeros(K,nbrOfSetups);
SE_OBE_DG_Distribued_Analytical_wo_PS_total = zeros(K,nbrOfSetups);

SE_OBE_DL_LSFD_Monte_wo_PS_total = zeros(K,nbrOfSetups);
SE_OBE_DL_Distribued_Monte_wo_PS_total = zeros(K,nbrOfSetups);
SE_OBE_DL_LSFD_Analytical_wo_PS_total = zeros(K,nbrOfSetups);
SE_OBE_DL_Distribued_Analytical_wo_PS_total = zeros(K,nbrOfSetups);

% % % % % % % ----Parallel computing
% core_number = 2;           
% parpool('local',core_number);
% % % % Starting parallel pool (parpool) using the 'local' profile ...

for i = 1:nbrOfSetups


    [R_AP,H_LoS_Single_real,channelGain,channelGain_LoS,channelGain_NLoS] = functionGenerateSetupDeploy(M,K,N,1,1);


    [H,H_LoS,PhaseMatrix] = functionChannelGeneration(R_AP,H_LoS_Single_real,M,K,N,nbrOfRealizations);
    [H_wo_PS,H_LoS_wo_PS] = functionChannelGeneration_wo_PS(R_AP,H_LoS_Single_real,M,K,N,nbrOfRealizations);


    A_singleLayer = reshape(repmat(eye(M),1,K),M,M,K);



    [Pset] = functionPilotAllocation(R_AP,H_LoS_Single_real,A_singleLayer,M,K,N,tau_p,pv);


    [Hhat_MMSE] = functionChannelEstimates_MMSE(R_AP,H_LoS,H,nbrOfRealizations,M,K,N,tau_p,pv,Pset);
    [Hhat_MMSE_wo_PS] = functionChannelEstimates_MMSE(R_AP,H_LoS_wo_PS,H_wo_PS,nbrOfRealizations,M,K,N,tau_p,pv,Pset);


    [Rhat,Phi,C_MMSE,C_total,C_total_blk] = functionMatrixGeneration(R_AP,pv,M,K,N,tau_p,Pset);



    %--Centralized

    %--Phase-shifts

    [V_MMSE_Combining] = functionMMSE_Combining_Centralized(Hhat_MMSE,C_total_blk,nbrOfRealizations,M,N,K,pv);
    [SE_MMSE_Centralized_Standard] = functionComputeSE_Centralized(Hhat_MMSE,V_MMSE_Combining,C_total_blk,tau_c,tau_p,nbrOfRealizations,M,N,K,pv);
    [SE_Centralized_MMSE_UatF] = functionComputeSE_Centralized_UatF_Monte(H,V_MMSE_Combining,tau_c,tau_p,nbrOfRealizations,M,N,K,pv);


    [V_OBE_Combining_Centralized,W_OBE_matrix_Centralized] = functionOBE_Combining_Centralized_Analytical(H_LoS_Single_real,Hhat_MMSE,Rhat,Phi,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);
    [SE_Centralized_UatF_OBE] = functionComputeSE_Centralized_UatF_Monte(H,V_OBE_Combining_Centralized,tau_c,tau_p,nbrOfRealizations,M,N,K,pv);
    [SE_Centralized_OBE_Monte_Standard] = functionComputeSE_Centralized(Hhat_MMSE,V_OBE_Combining_Centralized,C_total_blk,tau_c,tau_p,nbrOfRealizations,M,N,K,pv);

    [SE_Centralized_UatF_OBE_Analytical] = functionComputeSE_Centralized_UatF_Analytical(W_OBE_matrix_Centralized,H_LoS_Single_real,Rhat,R_AP,Phi,Pset,tau_c,tau_p,M,N,K,pv);

    %--Without Phase-shifts


    [V_MMSE_Combining_wo_PS] = functionMMSE_Combining_Centralized(Hhat_MMSE_wo_PS,C_total_blk,nbrOfRealizations,M,N,K,pv);
    [SE_MMSE_Centralized_wo_PS] = functionComputeSE_Centralized(Hhat_MMSE_wo_PS,V_MMSE_Combining_wo_PS,C_total_blk,tau_c,tau_p,nbrOfRealizations,M,N,K,pv);
    [SE_Centralized_MMSE_UatF_wo_PS] = functionComputeSE_Centralized_UatF_Monte(H_wo_PS,V_MMSE_Combining_wo_PS,tau_c,tau_p,nbrOfRealizations,M,N,K,pv);


    [V_OBE_Combining_Centralized_wo_PS,W_OBE_matrix_Centralized_wo_PS] = functionOBE_Combining_Centralized_Analytical_wo_PS(H_LoS_wo_PS,Hhat_MMSE_wo_PS,Rhat,Phi,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);
    [SE_Centralized_UatF_OBE_wo_PS] = functionComputeSE_Centralized_UatF_Monte(H_wo_PS,V_OBE_Combining_Centralized_wo_PS,tau_c,tau_p,nbrOfRealizations,M,N,K,pv);
    [SE_Centralized_OBE_Monte_Standard_wo_PS] = functionComputeSE_Centralized(Hhat_MMSE_wo_PS,V_OBE_Combining_Centralized_wo_PS,C_total_blk,tau_c,tau_p,nbrOfRealizations,M,N,K,pv);

    [SE_Centralized_OBE_UatF_Analytical_wo_PS] = functionComputeSE_Centralized_UatF_Analytical_wo_PS(W_OBE_matrix_Centralized_wo_PS,H_LoS_wo_PS,Rhat,R_AP,Phi,Pset,tau_c,tau_p,M,N,K,pv);


    %---Distributed



    %--Phase-shifts

    [V_MMSE_Combining_Distributed] = functionMMSE_Combining_Distributed(Hhat_MMSE,C_total,nbrOfRealizations,M,N,K,pv);
    [SE_MMSE_LSFD,SE_MMSE_Distributed] = functionComputeSE_Distributed_Monte(H,V_MMSE_Combining_Distributed,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);

    [V_OBE_Combining_Distributed,W_OBE_matrix_Distributed] = functionOBE_Combining_Distributed(H_LoS_Single_real,Hhat_MMSE,Rhat,Phi,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);
    [SE_OBE_Monte_LSFD,SE_OBE_Monte_Distributed] = functionComputeSE_Distributed_Monte(H,V_OBE_Combining_Distributed,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);


    [SE_OBE_Analytical_LSFD] = functionComputeSE_LSFD_Analytical(W_OBE_matrix_Distributed,H_LoS_Single_real,Rhat,R_AP,Phi,tau_c,tau_p,Pset,N,K,M,pv);
    [SE_OBE_Analytical_Distributed] = functionComputeSE_Distributed_Analytical(W_OBE_matrix_Distributed,H_LoS_Single_real,Rhat,R_AP,Phi,tau_c,tau_p,Pset,N,K,M,pv);


    [V_OBE_Combining_Distributed_Local,W_OBE_matrix_local] = functionOBE_Combining_Distributed_Local(H_LoS_Single_real,Hhat_MMSE,Rhat,Phi,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);
    [SE_OBE_Local_Monte_LSFD,SE_Local_OBE_Monte_Distributed] = functionComputeSE_Distributed_Monte(H,V_OBE_Combining_Distributed_Local,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);

    [SE_OBE_Local_Analytical_LSFD] = functionComputeSE_LSFD_Analytical(W_OBE_matrix_local,H_LoS_Single_real,Rhat,R_AP,Phi,tau_c,tau_p,Pset,N,K,M,pv);
    [SE_OBE_Local_Analytical_Distributed] = functionComputeSE_Distributed_Analytical(W_OBE_matrix_local,H_LoS_Single_real,Rhat,R_AP,Phi,tau_c,tau_p,Pset,N,K,M,pv);


    %--Without Phase-shifts

    [V_MMSE_Combining_Distributed_wo_PS] = functionMMSE_Combining_Distributed(Hhat_MMSE_wo_PS,C_total,nbrOfRealizations,M,N,K,pv);
    [SE_MMSE_LSFD_wo_PS,SE_MMSE_Distributed_wo_PS_wo_PS] = functionComputeSE_Distributed_Monte(H_wo_PS,V_MMSE_Combining_Distributed_wo_PS,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);

    %-DG-OBE

    [V_OBE_Combining_Distributed_wo_PS,W_OBE_matrix_Distributed_wo_PS] = functionOBE_Combining_Distributed_wo_PS(H_LoS_wo_PS,Hhat_MMSE_wo_PS,Rhat,Phi,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);
    [SE_OBE_Monte_LSFD_wo_PS,SE_OBE_Monte_Distributed_wo_PS] = functionComputeSE_Distributed_Monte(H_wo_PS,V_OBE_Combining_Distributed_wo_PS,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);

    [SE_OBE_Analytical_LSFD_wo_PS] = functionComputeSE_LSFD_Analytical_wo_PS(W_OBE_matrix_Distributed_wo_PS,H_LoS_wo_PS,Rhat,R_AP,Phi,tau_c,tau_p,Pset,N,K,M,pv);
    [SE_OBE_Analytical_Distributed_wo_PS] = functionComputeSE_Distributed_Analytical_wo_PS(W_OBE_matrix_Distributed_wo_PS,H_LoS_wo_PS,Rhat,R_AP,Phi,tau_c,tau_p,Pset,N,K,M,pv);



    %-DL-OBE
    [V_OBE_Combining_Distributed_Local_wo_PS,W_OBE_matrix_local_wo_PS] = functionOBE_Combining_Distributed_Local_wo_PS(H_LoS_wo_PS,Hhat_MMSE_wo_PS,Rhat,Phi,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);
    [SE_OBE_Local_Monte_LSFD_wo_PS,SE_Local_OBE_Monte_Distributed_wo_PS] = functionComputeSE_Distributed_Monte(H_wo_PS,V_OBE_Combining_Distributed_Local_wo_PS,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);

    [SE_Local_OBE_Analytical_LSFD_wo_PS] = functionComputeSE_LSFD_Analytical_wo_PS(W_OBE_matrix_local_wo_PS,H_LoS_wo_PS,Rhat,R_AP,Phi,tau_c,tau_p,Pset,N,K,M,pv);
    [SE_Local_OBE_Analytical_Distributed_wo_PS] = functionComputeSE_Distributed_Analytical_wo_PS(W_OBE_matrix_local_wo_PS,H_LoS_wo_PS,Rhat,R_AP,Phi,tau_c,tau_p,Pset,N,K,M,pv);




    SE_MMSE_Centralized_Standard_with_PS_total(:,i) = SE_MMSE_Centralized_Standard;
    SE_MMSE_Centralized_UatF_with_PS_total(:,i) = SE_Centralized_MMSE_UatF;
    SE_OBE_Centralized_UatF_Monte_with_PS_total(:,i) = SE_Centralized_UatF_OBE;
    SE_OBE_Centralized_OBE_Standard_Monte_with_PS_total(:,i) = SE_Centralized_OBE_Monte_Standard;
    SE_OBE_Centralized_UatF_Analytical_with_PS_total(:,i) = SE_Centralized_UatF_OBE_Analytical;

    SE_MMSE_Centralized_Standard_wo_PS_total(:,i) = SE_MMSE_Centralized_wo_PS;
    SE_MMSE_Centralized_UatF_wo_PS_total(:,i) = SE_Centralized_MMSE_UatF_wo_PS;
    SE_OBE_Centralized_UatF_Monte_wo_PS_total(:,i) = SE_Centralized_UatF_OBE_wo_PS;
    SE_OBE_Centralized_OBE_Standard_Monte_wo_PS_total(:,i) = SE_Centralized_OBE_Monte_Standard_wo_PS;
    SE_OBE_Centralized_UatF_Analytical_wo_PS_total(:,i) = SE_Centralized_OBE_UatF_Analytical_wo_PS;



    SE_LMMSE_LSFD_Monte_with_PS_total(:,i) = SE_MMSE_LSFD;

    SE_OBE_DG_LSFD_Monte_with_PS_total(:,i) = SE_OBE_Monte_LSFD;
    SE_OBE_DG_Distribued_Monte_with_PS_total(:,i) = SE_OBE_Monte_Distributed;
    SE_OBE_DG_LSFD_Analytical_with_PS_total(:,i) = SE_OBE_Analytical_LSFD;
    SE_OBE_DG_Distribued_Analytical_with_PS_total(:,i) = SE_OBE_Analytical_Distributed;


    SE_OBE_DL_LSFD_Monte_with_PS_total(:,i) = SE_OBE_Local_Monte_LSFD;
    SE_OBE_DL_Distribued_Monte_with_PS_total(:,i) = SE_Local_OBE_Monte_Distributed;
    SE_OBE_DL_LSFD_Analytical_with_PS_total(:,i) = SE_OBE_Local_Analytical_LSFD;
    SE_OBE_DL_Distribued_Analytical_with_PS_total(:,i) = SE_OBE_Local_Analytical_Distributed;

    SE_LMMSE_LSFD_Monte_wo_PS_total(:,i) = SE_MMSE_LSFD_wo_PS;

    SE_OBE_DG_LSFD_Monte_wo_PS_total(:,i) = SE_OBE_Monte_LSFD_wo_PS;
    SE_OBE_DG_Distribued_Monte_wo_PS_total(:,i) = SE_OBE_Monte_Distributed_wo_PS;
    SE_OBE_DG_LSFD_Analytical_wo_PS_total(:,i) = SE_OBE_Analytical_LSFD_wo_PS;
    SE_OBE_DG_Distribued_Analytical_wo_PS_total(:,i) = SE_OBE_Analytical_Distributed_wo_PS;


    SE_OBE_DL_LSFD_Monte_wo_PS_total(:,i) = SE_OBE_Local_Monte_LSFD_wo_PS;
    SE_OBE_DL_Distribued_Monte_wo_PS_total(:,i) = SE_Local_OBE_Monte_Distributed_wo_PS;

    SE_OBE_DL_LSFD_Analytical_wo_PS_total(:,i) = SE_Local_OBE_Analytical_LSFD_wo_PS;
    SE_OBE_DL_Distribued_Analytical_wo_PS_total(:,i) = SE_Local_OBE_Analytical_Distributed_wo_PS;




    disp([num2str(i) ' setups out of ' num2str(nbrOfSetups)]);



end




toc


 