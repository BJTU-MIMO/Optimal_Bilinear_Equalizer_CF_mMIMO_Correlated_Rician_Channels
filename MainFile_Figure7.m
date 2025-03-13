%%=============================================================
%The file is used to generates the data applied in Fig. 7 of the paper:
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

M = 40;
N_total = 1:1:6;
K = 20;
nbrOfSetups = 50;

nbrOfRealizations = 1000;

tau_p = 1;
tau_c = 200;

%Uplink transmit power per UE (W)
p = 0.2; %200 mW
%Create the power vector for all UEs (The uplink power is the same
%(p)at each UE)
pv = p*ones(1,K);



SE_OBE_Distributed_Monte_local_total = zeros(K,nbrOfSetups,length(N_total));
SE_OBE_Distributed_Analytical_local_total = zeros(K,nbrOfSetups,length(N_total));

SE_OBE_LSFD_Monte_local_total = zeros(K,nbrOfSetups,length(N_total));
SE_OBE_LSFD_Analytical_local_total = zeros(K,nbrOfSetups,length(N_total));


SE_OBE_Distributed_Monte_total = zeros(K,nbrOfSetups,length(N_total));
SE_OBE_Distributed_Analytical_total = zeros(K,nbrOfSetups,length(N_total));


SE_LMMSE_LSFD_total = zeros(K,nbrOfSetups,length(N_total));
SE_MR_LSFD_Monte_total = zeros(K,nbrOfSetups,length(N_total));
SE_MR_LSFD_Analytical_total = zeros(K,nbrOfSetups,length(N_total));




% % % % % % % ----Parallel computing
core_number = 2;           
parpool('local',core_number);
% % % Starting parallel pool (parpool) using the 'local' profile ...


for m = 1:length(N_total)

    N = N_total(m);
    

    parfor i = 1:nbrOfSetups
        
        [R_AP,H_LoS_Single_real,channelGain,channelGain_LoS,channelGain_NLoS] = functionGenerateSetupDeploy(M,K,N,1,1);
        [H,H_LoS] = functionChannelGeneration(R_AP,H_LoS_Single_real,M,K,N,nbrOfRealizations);
        
        A_singleLayer = reshape(repmat(eye(M),1,K),M,M,K);
        
        [Pset] = functionPilotAllocation(R_AP,H_LoS_Single_real,A_singleLayer,M,K,N,tau_p,pv);


        [Hhat_MMSE] = functionChannelEstimates_MMSE(R_AP,H_LoS,H,nbrOfRealizations,M,K,N,tau_p,pv,Pset);


        [Rhat,Phi,C_MMSE,C_total,C_total_blk] = functionMatrixGeneration(R_AP,pv,M,K,N,tau_p,Pset);



        % LMMSE combining
        [V_MMSE_Combining_Distributed] = functionMMSE_Combining_Distributed(Hhat_MMSE,C_total,nbrOfRealizations,M,N,K,pv);
        [SE_LMMSE_LSFD,~] = functionComputeSE_Distributed_Monte(H,V_MMSE_Combining_Distributed,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);


        % OBE
        [V_OBE_Combining_Distributed,W_OBE_matrix_Distributed] = functionOBE_Combining_Distributed(H_LoS_Single_real,Hhat_MMSE,Rhat,Phi,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);
        [~,SE_OBE_Distributed_Monte] = functionComputeSE_Distributed_Monte(H,V_OBE_Combining_Distributed,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);

        


        % Local OBE
        [V_OBE_Combining_Distributed_Local,W_OBE_matrix_Distributed_Local] = functionOBE_Combining_Distributed_Local(H_LoS_Single_real,Hhat_MMSE,Rhat,Phi,R_AP,Pset,M,N,K,pv,tau_p,nbrOfRealizations);
        [SE_OBE_Local_LSFD_Monte,SE_OBE_Local_Distributed_Monte] = functionComputeSE_Distributed_Monte(H,V_OBE_Combining_Distributed_Local,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);



        [SE_MR_LSFD_Monte,~] = functionComputeSE_Distributed_Monte(H,Hhat_MMSE,tau_c,tau_p,nbrOfRealizations,N,K,M,pv);


        [SE_OBE_Distributed_Analytical] = functionComputeSE_Distributed_Analytical(W_OBE_matrix_Distributed,H_LoS_Single_real,Rhat,R_AP,Phi,tau_c,tau_p,Pset,N,K,M,pv);
        [SE_OBE_Analytical_Distributed_Local] = functionComputeSE_Distributed_Analytical(W_OBE_matrix_Distributed_Local,H_LoS_Single_real,Rhat,R_AP,Phi,tau_c,tau_p,Pset,N,K,M,pv);
        [SE_OBE_Analytical_Local_LSFD] = functionComputeSE_LSFD_Analytical(W_OBE_matrix_Distributed_Local,H_LoS_Single_real,Rhat,R_AP,Phi,tau_c,tau_p,Pset,N,K,M,pv);


        W_MR_Dis = zeros(N,N,M,K);

        for k = 1:K
            for mm = 1:M
                W_MR_Dis(:,:,mm,k) = eye(N);

            end
        end

        [SE_MR_Analytical_LSFD] = functionComputeSE_LSFD_Analytical(W_MR_Dis,H_LoS_Single_real,Rhat,R_AP,Phi,tau_c,tau_p,Pset,N,K,M,pv);




 
        SE_OBE_Distributed_Monte_local_total(:,i,m) = SE_OBE_Local_Distributed_Monte;
        SE_OBE_Distributed_Monte_total(:,i,m) = SE_OBE_Distributed_Monte;


        SE_LMMSE_LSFD_total(:,i,m) = SE_LMMSE_LSFD;
        SE_OBE_LSFD_Monte_local_total(:,i,m) = SE_OBE_Local_LSFD_Monte;
        SE_MR_LSFD_Monte_total(:,i,m) = SE_MR_LSFD_Monte;

        SE_OBE_LSFD_Analytical_local_total(:,i,m) = SE_OBE_Analytical_Local_LSFD;
        SE_MR_LSFD_Analytical_total(:,i,m) = SE_MR_Analytical_LSFD;

        SE_OBE_Distributed_Analytical_local_total(:,i,m) = SE_OBE_Analytical_Distributed_Local;
        SE_OBE_Distributed_Analytical_total(:,i,m) = SE_OBE_Distributed_Analytical;



        disp([num2str(i) ' setups out of ' num2str(nbrOfSetups)]); 



    end
end



        

        
        

        
        
        
toc




 