%% BATCH OFFLINE 

clear all
rng default

R = 10;  % round
T = 1e3; % timestep
K = 5;   % number of branch
target = K; % il targhet è sempre l'ultima braccio
ind = ones(K,1); % funzione indicatrice
ind(target) = 0;

opt.sigma = 0.1;  % e-greedy
opt.eps = 0.1;    % e-greedy tradeoff (static)
opt.delta = 0.05; % attack
opt.xi = 0.001;
opt.opt = optimoptions('quadprog','Display','none');
opt.mu = [rand(1,K-1) 0]; % random expected values U(0,1);
opt.type = 'e-greedy';
opt.opt = optimoptions('quadprog','Display','none');


Cost = [];
M = 10:10:500; % size of branch
for m = M
    t = floor(T/m);
    opt.M = m;
    if m == 100
        oldCnt = NaN(t,K);
    end
    avgOld = NaN(t,K);
    cost = NaN(t,1);
    chose = NaN(t,1);
    avg = zeros(1,K);
    cnt = ones(1,K);

    for b = 1:t
        % clacolo del batch
        [cnt,avg] = computeBatch(m,avg,cnt,opt);
        if m == 100
            oldCnt(b,:)= cnt;
        end
        %attacco
        [avg c] = attack(cnt,avg,[],opt);
        avgOld(b,:)= avg; 
        cost(b,1) = c;
        [~,c] = max(avg);
        chose(b,1) = c; 
    end
    
    if m == 100
        pa = sum(cost);
    end
    Cost = [Cost;sum(cost)];
end


figure
b = bar(oldCnt);
legend(["a_1","a_2","a_3","a_4","a^*"])
xlabel("i-batch")
ylabel("Pulled arm")
title("Size-Batch: " + 100 + " Costo: " + pa)

% savefig("batched_e-greddy");
% print("batched_e-greddy",'-dpng');


figure
plot(Cost)
title("Costo al variare della grandezza del batch");
xticks([1:size(Cost,1)])
xticklabels(string(M))
xtickangle(90)
ylabel("Costo")
xlabel("Batch-size")

% savefig("batchet_cost_e-greedy");
% print("batchet_cost_e-greedy",'-dpng');

