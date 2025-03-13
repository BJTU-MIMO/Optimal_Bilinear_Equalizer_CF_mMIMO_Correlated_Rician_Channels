%%=============================================================
%The file is used to generates the data applied in Fig. 2 of the paper:
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

nbrOfSetups = 200;

nbrOfRealizations = 1000;

tau_p = 1;
tau_c = 200;

%Uplink transmit power per UE (W)
p = 0.2; %200 mW
%Create the power vector for all UEs (The uplink power is the same
%(p)at each UE)
pv = p*ones(1,K);

SE_Centralized_Standard_MMSE = zeros(K,nbrOfSetups,length(M));
SE_Centralized_UatF_MMSE = zeros(K,nbrOfSetups,length(M));
SE_Centralized_Genie_Aided_MMSE_Monte = zeros(K,nbrOfSetups,length(M));

SE_Centralized_Standard_OBE_Monte = zeros(K,nbrOfSetups,length(M));
SE_Centralized_UatF_OBE_Monte = zeros(K,nbrOfSetups,length(M));
SE_Centralized_Genie_Aided_OBE_Monte = zeros(K,nbrOfSetups,length(M));



% % % % % % % ----Parallel computing
core_number = 2;           
parpool('local',core_number);
% % % Starting parallel pool (parpool) using the 'local' profile ...

for m = 1:length(M)
    parfor i = 1:nbrOfSetups
        
        [R_AP,H_LoS_Single_real,channelGain,channelGain_LoS,channelGain_NLoS] = functionGenerateSetupDeploy(M(m),K,N,1,1);
        [H,H_LoS] = functionChannelGeneration(R_AP,H_LoS_Single_real,M(m),K,N,nbrOfRealizations);
        
        A_singleLayer = reshape(repmat(eye(M(m)),1,K),M(m),M(m),K);
        
        [Pset] = functionPilotAllocation(R_AP,H_LoS_Single_real,A_singleLayer,M(m),K,N,tau_p,pv);


        [Hhat_MMSE] = functionChannelEstimates_MMSE(R_AP,H_LoS,H,nbrOfRealizations,M(m),K,N,tau_p,pv,Pset);


        [Rhat,Phi,C_MMSE,C_total,C_total_blk] = functionMatrixGeneration(R_AP,pv,M(m),K,N,tau_p,Pset);

        [V_MMSE_Combining] = functionMMSE_Combining_Centralized(Hhat_MMSE,C_total_blk,nbrOfRealizations,M,N,K,pv);
        [SE_Centralized_UatF_MMSE_i] = functionComputeSE_Centralized_UatF_Monte(H,V_MMSE_Combining,tau_c,tau_p,nbrOfRealizations,M,N,K,pv);
        [SE_Centralized_Standard_MMSE_i] = functionComputeSE_Centralized(Hhat_MMSE,V_MMSE_Combining,C_total_blk,tau_c,tau_p,nbrOfRealizations,M(m),N,K,pv);
        [SE_Centralized_Genie_Aided_MMSE_Monte_i] = functionComputeSE_Centralized_Genie_Aided(H,V_MMSE_Combining,tau_c,tau_p,nbrOfRealizations,M,N,K,pv);


        % DG-OBE can achieve similar performance with C-OBE
        [V_OBE_Combining_Distributed,~] = functionOBE_Combining_Distributed(H_LoS_Single_real,Hhat_MMSE,Rhat,Phi,R_AP,Pset,M(m),N,K,pv,tau_p,nbrOfRealizations);

        [SE_Centralized_Standard_OBE_i] = functionComputeSE_Centralized(Hhat_MMSE,V_OBE_Combining_Distributed,C_total_blk,tau_c,tau_p,nbrOfRealizations,M(m),N,K,pv);
        [SE_Centralized_UatF_OBE_Monte_i] = functionComputeSE_Centralized_UatF_Monte(H,V_OBE_Combining_Distributed,tau_c,tau_p,nbrOfRealizations,M(m),N,K,pv);
        [SE_Centralized_Genie_Aided_OBE_Monte_i] = functionComputeSE_Centralized_Genie_Aided(H,V_OBE_Combining_Distributed,tau_c,tau_p,nbrOfRealizations,M,N,K,pv);



        SE_Centralized_Standard_OBE_Monte(:,i,m) = SE_Centralized_Standard_OBE_i;
        SE_Centralized_UatF_OBE_Monte(:,i,m) = SE_Centralized_UatF_OBE_Monte_i;
        SE_Centralized_Genie_Aided_OBE_Monte(:,i,m) = SE_Centralized_Genie_Aided_OBE_Monte_i;
        

        SE_Centralized_Standard_MMSE(:,i,m) = SE_Centralized_Standard_MMSE_i;
        SE_Centralized_UatF_MMSE(:,i,m) = SE_Centralized_UatF_MMSE_i;
        SE_Centralized_Genie_Aided_MMSE_Monte(:,i,m) = SE_Centralized_Genie_Aided_MMSE_Monte_i;





        disp([num2str(i) ' setups out of ' num2str(nbrOfSetups)]); 



    end
end




toc




 