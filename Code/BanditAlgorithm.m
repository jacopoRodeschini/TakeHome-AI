
%% DATI DEL PROBLEMA
clear all
rng default

K = 5; % #arm
mu = [0.2 0.4 0.9 0.3 0];

mu(K)= 0;

T = 1000; % time-step
R = 1000;  % round 

%sigma = 1; 
sigma = 0.1;

eps = 0.1;

% il risultato migliore ottenibile 0.9 * 1000 = 900
% ovvero scelgo sempre la scelta 0.9. 

% l'agente migliore è quello che più si avvicina a 900.

res = [];    % performance algoritmi

%% AGENTE CASUALE
% teorico ott. = 360

choice = randi(K,[T,R]);
succ = NaN(T,K,R);

for j = 1:R
    for i = 1:K
         inx = find(choice(:,j) == i);
         % simulazione dell'ambiente in cui l'agente opera
         succ(inx,i,j) = mu(i) + randn*sigma; 
         
    end
end

% valore atteso 
res = [res nanmean(nansum(nansum(succ,1),2))] 

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
res = [res nanmean(value)]; % risultati

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

res = [res nanmean(value)];


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

res = [res nanmean(value)]; 

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

% valore atteso medio
res = [res nanmean(value)]

%% COMPARISON PERFORMANCE
figure
bar(res)
grid on 
grid minor
xticks([1:K])
xticklabels(["Casuale", "e-greedy", "e-greedy dec", "UCB", "Thompson sampling"])
xtickangle(45);

% savefig("comparins normal");
% print("comparins normal",'-dpng')
