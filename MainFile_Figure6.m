%%=============================================================
%The file is used to generates the data applied in Fig. 6 of the paper:
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
K_total = 20;
nbrOfSetups = 200;

nbrOfRealizations = 1000;

tau_p_total = [1,10];
tau_c = 200;

%Uplink transmit power per UE (W)
p = 0.2; %200 mW
%Create the power vector for all UEs (The uplink power is the same
%(p)at each UE)



SE_Distributed_LSFD_MMSE = zeros(K_total,nbrOfSetups,length(tau_p_total));
SE_Distributed_OBE_Monte = zeros(K_total,nbrOfSetups,length(tau_p_total));


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



            % LMMSE combining
            [V_MMSE_Combining_Distributed] = functionMMSE_Combining_Distributed(Hhat_MMSE,C_total,nbrOfRealizations,M,N,K,pv);
            [SE_LMMSE_LSFD,SE_LMMSE_Distributed] = functionComputeSE_Distributed_Monte(H,V_MMSE_Combining_Distributed,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);


            % DG-OBE
            [V_OBE_Combining_Distributed,W_OBE_matrix_Distributed] = functionOBE_Combining_Distributed(H_LoS_Single_real,Hhat_MMSE,Rhat,Phi,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);
            [SE_OBE_LSFD_Monte,SE_OBE_Distributed_Monte] = functionComputeSE_Distributed_Monte(H,V_OBE_Combining_Distributed,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);

            SE_Distributed_LSFD_MMSE(:,i,n) = SE_LMMSE_LSFD;
            SE_Distributed_OBE_Monte(:,i,n)  = SE_OBE_Distributed_Monte;




            disp([num2str(i) ' setups out of ' num2str(nbrOfSetups)]);
            disp([num2str(i) ' setups out of ' num2str(nbrOfSetups)]);
 


        end



    end
end




toc




 