clear all
rng default

K = 5; % #arm
%mu = [0.2 0.4 0.9 0.3 0];
mu = rand(1,K);
mu(K)= 0;

T = 1000; % time-step
R = 1000;  % round 

% il risultato migliore ottenibile 0.9 * 1000 = 900
% ovvero scelgo sempre la scelta 0.9. 

% l'agente migliore è quello che più si avvicina a 900.

res = [];    % performance algoritmi

sigma = .1;
opt.sigma = sigma;
opt.opt = optimoptions('quadprog','Display','none');
opt.ski = 0.001; % variabile di scarto / surplus
opt.delta = (K-1)/10; % <= (K-1)/2 

%% AGENTE E-GREEDY

eps = 0.1; 
prob = rand([T,R]);
chgreedy = NaN(T,K,R);
succgreedy = NaN(T,K,R);
value = NaN(R,1);
for j = 1:R
    for i = 1:T  
        if i <= K
            t = i; 
        else
            if prob(i,j) <=eps
                t = randi(K,1);
            else
                avg = nanmean(succgreedy(:,:,j));
                [m ,t] = nanmax(avg) ; % e-greedy
            end
        end
        
        % simulazione dell'embiente in cui l'agente opera
        chgreedy(i,t,j) = 1;
        succgreedy(i,t,j) = binornd(1,mu(t));
    end
    value(j,1) = nansum(nansum(succgreedy(:,:,j),1),2); 
end

% media stimatori
st = nanmean(nanmean(succgreedy),3); % stima buona per i brach con prob piu alta.
res = [res nanmean(value)] 
%stimaPar = [stimaPar st'];

[cost next] = meanBasedAttackGreedy(chgreedy,succgreedy,opt);

ratiogrbino = sqrt(cost./value);
tabgrbino = tabulate(next);

figure
subplot(2,1,1)
bar(tabgrbino(:,2))
title("Poisoning attacks over e-greedy e = 0.1 Binomial")
legend(["T+1 choice"],'Location','Best')
xlabel("i-brach")
ylabel("Number of trials")

subplot(2,1,2)
histogram(ratiogrbino)
title("Ratio histogram")
ylabel("Number of trials")
xlabel("Effor ratio")

% savefig("attacks_over_e-greedy_bino")
% print("attacks_over_e-greedy_bino","-dpng")



 %% AGENTE E-GREEDY DECRESCENTE 

prob = rand([T,R]);
ch = NaN(T,K,R);
succ = NaN(T,K,R);
value = NaN(R,1);
for j = 1:R
    for i = 1:T
        if i <= K
            t = i; 
        else
            eps = 1/t;
            if prob(i,j) <=eps
                t = randi(K,1);
            else
                avg = nanmean(succ(:,:,j));
                [m ,t] = nanmax(avg) ; % e-greedy
            end
        end
        
        % simulazione dell'embiente in cui l'agente opera
        ch(i,t,j) = 1;
        succ(i,t,j) = binornd(1,mu(t));
    end
    value(j,1) = nansum(nansum(succ(:,:,j),1),2); 
end



% media stimatori
st = nanmean(nanmean(succ),3); % stima buona per i brach con prob piu alta.
res = [res nanmean(value)]

[cost next] = meanBasedAttackGreedy(ch,succ,opt);

ratiogrdbino = sqrt(cost./value);
tabgrdbino = tabulate(next);

figure
subplot(2,1,1)
bar(tabgrdbino(:,2))
title("Poisoning attacks over e-greedy e=1/t Binomiale")
legend(["T+1 choice"],'Location','Best')
xlabel("i-brach")
ylabel("Number of trials")

subplot(2,1,2)
histogram(ratiogrdbino)
title("Ratio histogram")
ylabel("Number of trials")
xlabel("Effor ratio")

% savefig("attacks_over_e-greedy_dec_bino norm")
% print("attacks_over_e-greedy_dec_bino","-dpng")


%%  THOMPSON (Beta distribuiton)

succ = NaN(T,K,R);
for j = 1:R
    for i = 1:T
        s = nansum(succ(:,:,j)) + ones(1,K); % successi  
        f = nansum(succ(:,:,j)== 0) + ones(1,K); % fallimenti
        avg =  betarnd(s,f);  %return probability for choice
        [m ,t] = max(avg);
        succ(i,t,j) = binornd(1,mu(t));
    end
end

st = nanmean(nanmean(succ,1),3); % le piu alte sonostimate giuste

% valore atteso 
rr = nanmean(nansum(nansum(succ,1),2),3) ;%

res = [res rr];

%% RESULT COMPARISON (Beta distributions)
figure
bar(res)
grid on
grid minor
title("Performance Comparison")
ylabel("Expected value")
xticks([1:4])
xticklabels(["e-greedy e = 0.1" "e-greedy e = 1/t","Thompson Sampling"])
xtickangle(45)

% savefig("Performance_Comparison_Binomiale")
% print("Performance_Comparison_Binomiale","-dpng")

