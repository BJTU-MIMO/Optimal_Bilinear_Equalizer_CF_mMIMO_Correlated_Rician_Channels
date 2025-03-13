%%=============================================================
%The file is used to generates the data applied in Fig. 8 of the paper:
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


Rician_Region = [0.001,10:10:500,10000]';



SE_OBE_Monte_LSFD_Total = zeros(K,nbrOfSetups,length(Rician_Region));
SE_OBE_Monte_Distributed_Total = zeros(K,nbrOfSetups,length(Rician_Region));

SE_OBE_Analytical_LSFD_Total = zeros(K,nbrOfSetups,length(Rician_Region));
SE_OBE_Analytical_Distributed_Total = zeros(K,nbrOfSetups,length(Rician_Region));

SE_LMMSE_Monte_LSFD_Total = zeros(K,nbrOfSetups,length(Rician_Region));
SE_LMMSE_Monte_Distributed_Total = zeros(K,nbrOfSetups,length(Rician_Region));


A_singleLayer = reshape(repmat(eye(M),1,K),M,M,K);

% % % % % % % ----Parallel computing
core_number = 2;           
parpool('local',core_number);
% % % Starting parallel pool (parpool) using the 'local' profile ...

for n = 1:nbrOfSetups
    
    [R_AP_Total,HMean_Withoutphase_Total] = functionGenerateSetupDeploy_Rician_Regions(M,K,N,1,1,Rician_Region);
    
    parfor ii = 1:length(Rician_Region)

        R_AP = R_AP_Total(:,:,:,:,ii);
        H_LoS_Single_real = HMean_Withoutphase_Total(:,:,ii);




        [Pset] = functionPilotAllocation(R_AP,H_LoS_Single_real,A_singleLayer,M,K,N,tau_p,pv);

        [H,H_LoS] = functionChannelGeneration(R_AP,H_LoS_Single_real,M,K,N,nbrOfRealizations);



        [Hhat_MMSE] = functionChannelEstimates_MMSE(R_AP,H_LoS,H,nbrOfRealizations,M,K,N,tau_p,pv,Pset);


        [Rhat,Phi,C_MMSE,C_total,C_total_blk] = functionMatrixGeneration(R_AP,pv,M,K,N,tau_p,Pset);


        %---Distributed


        [V_OBE_Combining_Distributed,W_OBE_matrix_Distributed] = functionOBE_Combining_Distributed(H_LoS_Single_real,Hhat_MMSE,Rhat,Phi,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);

        [SE_OBE_Monte_LSFD,SE_OBE2_Monte] = functionComputeSE_Distributed_Monte(H,V_OBE_Combining_Distributed,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);

        [SE_OBE_Analytical] = functionComputeSE_Distributed_Analytical(W_OBE_matrix_Distributed,H_LoS_Single_real,Rhat,R_AP,Phi,tau_c,tau_p,Pset,N,K,M,pv);
        [SE_OBE_LSFD_Analytical] = functionComputeSE_LSFD_Analytical(W_OBE_matrix_Distributed,H_LoS_Single_real,Rhat,R_AP,Phi,tau_c,tau_p,Pset,N,K,M,pv);

        [V_MMSE_Combining_Distributed] = functionMMSE_Combining_Distributed(Hhat_MMSE,C_total,nbrOfRealizations,M,N,K,pv);
        [SE_LMMSE_Monte_LSFD,SE_LMMSE_Distributed_LSFD] = functionComputeSE_Distributed_Monte(H,V_MMSE_Combining_Distributed,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);



        SE_OBE_Monte_LSFD_Total(:,n,ii) = SE_OBE_Monte_LSFD;
        SE_OBE_Monte_Distributed_Total(:,n,ii) = SE_OBE2_Monte;
        SE_LMMSE_Monte_LSFD_Total(:,n,ii) = SE_LMMSE_Monte_LSFD;
        SE_LMMSE_Monte_Distributed_Total(:,n,ii) = SE_LMMSE_Distributed_LSFD;


        SE_OBE_Analytical_LSFD_Total(:,n,ii) = SE_OBE_LSFD_Analytical;
        SE_OBE_Analytical_Distributed_Total(:,n,ii) = SE_OBE_Analytical;
        
        disp([num2str(n) ' setups out of ' num2str(nbrOfSetups)]);
        disp([num2str(ii) ' -th Rician Region out of ' num2str(length(Rician_Region))]);
        



    end
    


end
toc


 