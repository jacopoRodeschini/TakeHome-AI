%% Offline: Normal Distributions
%  Suppose normal distributions of reward (mu are unknow)

%% DATI DEL PROBLEMA
tic
clear all
rng default

K = 5; % #arm

mu = rand(1,K);
mu(K)= 0;

T = 1000; % time-step
R = 1000;  % round 


sigma = .1;
opt.sigma = sigma;
opt.opt = optimoptions('quadprog','Display','none');
opt.ski = 0.001; % variabile di scarto / surplus
opt.delta = (K-1)/10; % <= (K-1)/2 (TS)
eps = 0.1; 

%% E-GREEDY 

prob = rand([T,R]);
ch = NaN(T,K,R);
succ = NaN(T,K,R);
value = NaN(R,1);
for j = 1:R
   % ch(1,:,j) = ones(1,K);
    for i = 1:T
        %eps = 1 / log(t); % epselon decreasing
        if i <= K
            t = i; 
        else
            %eps = 1/t;
            if prob(i,j) <=eps
                t = randi(K,1);
            else
                %cnt = nansum(ch(:,:,j));
                avg = nanmean(succ(:,:,j));
                [m ,t] = nanmax(avg) ; % e-greedy
            end
        end
        ch(i,t,j) = 1;
        
        % simulazione dell'embiente in cui l'agente opera
        succ(i,t,j) = mu(t) + randn*sigma;
    end
    value(j,1) = nansum(nansum(succ(:,:,j),1),2); 
end

% senza fare l'attacco:
chne = [];
for i = 1:R
    [m,chne] = max(nanmean(succ(:,:,j)));
end
tab = tabulate(chne);

figure
bar(tab(:,2));
xticks([1:5]);
xticklabels(mu);
legend(["t+1 choice"])
title("Frequenza delle scelte senza attacco")
ylabel("Number of trials")
xlabel("i-arms")

savefig("without_attack")
print("without_attack","-dpng")


%attaco
[cost next] = meanBasedAttackGreedy(ch,succ,opt);


ratiogr = sqrt(cost./value);
tabgr = tabulate(next);

figure
subplot(2,1,1)
bar(tabgr(:,2))
title("Poisoning attacks over e-greedy normal")
legend(["T+1 choice"],'Location','Best')
xlabel("i-arms")
ylabel("Number of trials")

subplot(2,1,2)
histogram(ratiogr)
title("Ratio histogram")
ylabel("Number of trials")
xlabel("Effor ratio")

savefig("attacks_over_e-greedy_norm")
print("attacks_over_e-greedy_norm","-dpng")

%% E-GREEDY DEC 

prob = rand([T,R]);
succ = NaN(T,K,R);
ch = NaN(T,K,R);
value = NaN(R,1);
for j = 1:R
   % ch(1,:,j) = ones(1,K);
    for i = 1:T
        if i <= K
            t = i; 
        else
            eps = 1/t;
            if prob(i,j) <=eps
                t = randi(K,1);
            else
                %cnt = nansum(ch(:,:,j));
                avg = nanmean(succ(:,:,j));
                [m ,t] = nanmax(avg) ; % e-greedy
            end
        end
        ch(i,t,j) = 1;
        % simulazione dell'embiente in cui l'agente opera
        succ(i,t,j) = mu(t) + randn*sigma;
    end
    value(j,1) = nansum(nansum(succ(:,:,j),1),2); 
end

[cost next] = meanBasedAttackGreedy(ch,succ,opt);

ratiogrd = sqrt(cost./value);
tabgrd = tabulate(next);

figure
subplot(2,1,1)
bar(tabgrd(:,2))
title("Poisoning attacks over e-greedy dec. normal")
legend(["T+1 choice"],'Location','Best')
xlabel("i-arms")
ylabel("Number of trials")

subplot(2,1,2)
histogram(ratiogrd)
title("Ratio histogram")
ylabel("Number of trials")
xlabel("Effor ratio")

savefig("attacks_over_e-greedy_dec_norm")
print("attacks_over_e-greedy_dec_norm","-dpng")



%%  UCB (Upper Confident Bound )

ch = NaN(T,K,R);
succ = NaN(T,K,R);
value = NaN(R,1);
for j = 1:R
    for i = 1:T
        if i<=K
            t = i;
        else
            cnt = nansum(ch(:,:,j));
            avg = nanmean(succ(:,:,j));    
            [m ,t] = max(avg + 3*sigma*sqrt(log(t) ./cnt ) ) ; % ucb
        end
        ch(i,t,j) = 1;
        succ(i,t,j) = mu(t) + randn*sigma;
    end
    value(j,1) = nansum(nansum(succ(:,:,j),1),2); 
end

% attacco
[cost next] = meanBasedAttackUCB(ch,succ,opt);

ratioucb = sqrt(cost./value);
tabucb = tabulate(next);

figure
subplot(2,1,1)
bar(tabucb(:,2))
title("Poisoning attacks over UCB normal")
legend(["T+1 choice"],'Location','Best')
xlabel("i-arms")
ylabel("Number of trials")

subplot(2,1,2)
histogram(ratioucb)
title("Ratio histogram")
ylabel("Number of trials")
xlabel("Effor ratio")
 
savefig("attacks_over_ucb_norm")
print("attacks_over_ucb_norm","-dpng")


%% THOMPSON SAMPLING

succ = NaN(T,K,R);
ch = NaN(T,K,R);
for j = 1:R
    ch(1:K,:,j) = eye(K);
    for i = 1:T
        if i<=K
            t = i;
        else
            cnt = nansum(ch(:,:,j));    % push branch  
            avg = nanmean(succ(:,:,j)); % mean branch 
            avg(isnan(avg)) = 0;
            [m ,t] = max(avg/(sigma^2) + randn*sqrt(sigma^2./cnt));
        end
        ch(i,t,j) = 1;
        succ(i,t,j) = mu(t) + randn*sigma;
    end
    value(j) = nansum(nansum(succ(:,:,j),1),2); 
end

[cost, ch_next] = meanBasedAttackThompson(ch,succ,opt);

ratioth = sqrt(cost./value);
tabth = tabulate(next);

figure
subplot(2,1,1)
bar(tabth(:,2))
title("Poisoning attacks over Thompson normal")
legend(["T+1 choice"],'Location','Best')
xlabel("i-arms")
ylabel("Number of trials")

subplot(2,1,2)
histogram(ratioth)
title("Ratio histogram")
ylabel("Number of trials")
xlabel("Effor ratio")

savefig("attacks_over_thompson_norm")
print("attacks_over_thompson_norm","-dpng")

%% END 
toc



