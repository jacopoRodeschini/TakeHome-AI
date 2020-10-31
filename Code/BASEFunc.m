tic
clear all 
rng default

K = 5; 
M = 10; % numero di batch 
T = 1e3;

opt.sigma = 0.1;  % e-greedy
opt.eps = 0.1;    % e-greedy tradeoff (static)
opt.delta = 0.05; % attack
opt.xi = 0.001;
opt.mu = [rand(1,K-1) 0]; % random expected values U(0,1);
opt.gamma = 0.1;
opt.type = 'e-greedy';
opt.opt = optimoptions('quadprog','Display','none');
opt.T = T;
opt.K = K;



a = T^(1/(2 - 2^(1-M))); 
TGrid = floor(a.^(2.-1./2.^(0:M-1)));...,
TGrid(M) = T; 
TGrid = [0,TGrid]; % minimax batch grids
regret = 0;

% initialization
activeSet = ones(1,K); 
numberPull = zeros(1,K); 
averageReward = zeros(1,K);
Cost = 0;
avgOld = zeros(1,K);
for i = 2:M+1
    availableK = sum(activeSet);
    pullNumber = max(floor((TGrid(i) - TGrid(i-1))/availableK), 1);
    TGrid(i) = availableK * pullNumber + TGrid(i-1);
    opt.M = availableK * pullNumber;
    inx = find(activeSet == 1);

    k = size(inx,2);
    % tira il braccio "pullNumber" volte 
    avg = mean(randn(k,pullNumber)*opt.sigma,2)' + opt.mu(inx); % media reward dopo pullNumeber tiri. 

    % attacco BaSE 
    opt.tot = numberPull(end) + pullNumber; 
    opt.rho = numberPull(inx)./opt.tot;
    cnt = ones(1,k)*pullNumber;
    
    [avg c] = attack(cnt,avg,averageReward(inx),opt);
    %Cost = Cost + c; 
    
    averageReward(inx) =  averageReward(inx).*opt.rho + avg.*(1-opt.rho);
    
    % reget con attacco 
    regret = regret + (pullNumber*ones(1,k) * (opt.mu(1) - opt.mu(inx))');
    numberPull(inx) = numberPull(inx) + pullNumber; 

    maxArm = max(averageReward(inx));
    activeSet(inx) = (maxArm - averageReward(inx)) < sqrt(opt.gamma * log(opt.T*opt.K) ./ opt.tot)   
end

activeSet
Cost
toc
    