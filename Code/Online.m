
%% ACE ONLINE ATTACK
clear all
rng default

R = 10;  % round
T = 1e4; % timestep
K = 5;    % number of branch
target = K; % il targhet è sempre l'ultima braccio
ind = ones(K,1); % funzione indicatrice
ind(target) = 0;

sigma = 0.1;  % e-greedy
eps = 0.1;    % e-greedy tradeoff (static)
delta = 0.05; % attack

% universal B(n) functions
beta = @(n) sqrt(2*sigma^2/n*log(pi^2*n*K^2/(3*delta)));

mu = [rand(1,K-1) 0]; % random expected values U(0,1);


%% e-greedy (Normal Distributios)
cumErr = [];
cumTar = [];
cumTarNot = [];


for flag = 0:1
    rewa = NaN(T,K,R); % reward
    rewaNot = NaN(T,K,R); % reward in caso di non attacco 
    c = NaN(T,R); % costo di attacco 
    Target = NaN(T,K);
    TargetNot = NaN(T,K); % target in caso di non attacco
    for r = 1:R
        cnt = zeros(1,K);
        %avg = zeros(1,K);
        avgU = zeros(1,K); % avarege reward non modificati R(eal)
        for t = 1:T
            if t<=K
                dec = t;
                decNot = dec; 
            else
                if flag
                    eps = 1/t;
                end
                if rand < eps
                    dec = randi(K);
                    decNot = dec;
                else
                    avg = nanmean(rewa(:,:,r));
                    [m,dec] = nanmax(avg);
                    avg = nanmean(rewaNot(:,:,r));
                    [m,decNot] = nanmax(avg);
                end
            end
            % expected reward before attack 
            rd = randn;

            rw = mu(dec) + rd*sigma; % r = u + noise (Normal)
            rwn = mu(decNot) + rd*sigma;
            

            % media dei reward unbiased 
            avgU(dec) = (avgU(dec)*cnt(dec) + rw)/(cnt(dec)+1);

            cnt(dec) = cnt(dec)+1;
            if t <= K
                att = 0; % non esguo nessun attocco
            else
                att = -ind(dec) *nanmax(0,avgU(dec) - avgU(K) ...
                          + beta(cnt(dec)) + beta(cnt(K)));
            end
            c(t,r) = abs(att); % norma 1
            rewa(t,dec,r) = rw + att;
            %avg(dec) = (avg(dec)*(cnt(dec)-1) + rw  + att)/cnt(dec); 
            Target(t,r) = dec;

            % se non ci fose stato l'attacco: 
            rewaNot(t,decNot,r) = rwn;
            TargetNot(t,r) = decNot; 
        end
    end
   Target = Target == target;
   TargetNot = TargetNot == target;
   
   cumTarNot = [cumTarNot cumsum(nanmean(TargetNot,2))]; 
   
   cumErr = [cumErr cumsum(nanmean(c,2))];
   cumTar = [cumTar cumsum(nanmean(Target,2))]; 
end



figure
subplot(2,1,1)
loglog([1:T],[cumErr(:,1) cumErr(:,2)]);
grid on
title("Costo cumulato e-greedy Normale")
ylabel("Costo")
xlabel("Time step")
legend(["e-greedy","e-greedy dec."],'Location','Best')

subplot(2,1,2)
plot([1:T],[cumTar(:,1) cumTar(:,2)]);
hold on 
plot([1:T],[cumTarNot(:,1) cumTarNot(:,2)],'g');
grid on 
grid minor
title("#Pulled arm a^*")
ylabel("Pulled arm")
xlabel("Time step")
legend(["ACE Attack","ACE attack e-greedy dec.","No Attack","No Attack e-greedy dec"],'Location','Best')

% savefig("online_e_greddy_norm")
% print("online_e_greddy_norm",'-dpng')


%% e-greedy (Binomial Distributios)


rewa = NaN(T,K,R); % reward
rewaNot = NaN(T,K,R); % reward
c = NaN(T,R); % costo di attacco 
Target = NaN(T,K);
TargetNot = NaN(T,K);

for r = 1:R
    cnt = zeros(1,K);
    %avg = zeros(1,K);
    avgU = zeros(1,K); % avarege reward non modificati U(nbiased)
    for t = 1:T
        if t<=K
            dec = t;
            decNot = dec;
        else
            if rand < eps
                dec = randi(K);
                decNot = dec;
            else
                avg = nanmean(rewa(:,:,r));
                [m,dec] = nanmax(avg);
                avg = nanmean(rewaNot(:,:,r));
                [m,decNot] = nanmax(avg);
            end
        end
        % expected reward before attack 
        rw = binornd(1,mu(dec));
        rwn = binornd(1,mu(decNot));
        
        % media dei reward unbiased 
        avgU(dec) = (avgU(dec)*cnt(dec) + rw)/(cnt(dec)+1);

        cnt(dec) = cnt(dec)+1;
        if t <= K
            att = 0; % non esguo nessun attacco
        else
            att = -ind(dec) *nanmax(0,avgU(dec) - avgU(K) ...
                      + beta(cnt(dec)) + beta(cnt(K)));
        end
        c(t,r) = abs(att); % norma 1
        rewa(t,dec,r) = rw + att;
        Target(t,r) = dec;
        
        rewaNot(t,decNot,r) = rw;
        TargetNot(t,r) = decNot;
    end
end

Target = Target == target;
TargetNot = TargetNot == target; 

figure
subplot(2,1,1)
semilogx([1:T],cumsum(nanmean(c,2)));
grid on
title("Costo cumulato e-greedy Binomiale")
ylabel("Costo")
xlabel("Time step")
legend(["e-greedy bino"],'Location','Best')

subplot(2,1,2)
plot([1:T],cumsum(nanmean(Target,2)));
hold on 
plot([1:T],cumsum(nanmean(TargetNot,2)),'g');
grid on
grid minor
title("#Pulled arm a^*")
ylabel("Pulled arm")
xlabel("Time step")
legend(["ACE Attack","No Attack"],'Location','Best')

% savefig("online_e_greddy_bino")
% print("online_e_greddy_bino",'-dpng')

%% UBC Normal

rewa = NaN(T,K,R); % reward
rewaNot = NaN(T,K,R); % reward in caso di non attacco 
c = NaN(T,R); % costo di attacco 
Target = NaN(T,K);
TargetNot = NaN(T,K); % target in caso di non attacco
for r = 1:R
    cnt = zeros(1,K);
    cntNot = zeros(1,K);
    avgU = zeros(1,K); % avarege reward non modificati R(eal)
    for t = 1:T
        if t<=K
            dec = t;
            decNot = dec; 
        else  
            avg = nanmean(rewa(:,:,r));
            [m,dec] = nanmax(avg + 3*sigma*sqrt(log(t)./cnt));

            avg = nanmean(rewaNot(:,:,r));
            [m,decNot] = nanmax(avg + 3*sigma*sqrt(log(t)./cntNot));       
        end
        % expected reward before attack
        rd = randn;
        rw = mu(dec) + rd*sigma; % r = u + noise (Normal)
        rwn = mu(decNot) + rd*sigma;

        % media dei reward unbiased 
        avgU(dec) = (avgU(dec)*cnt(dec) + rw)/(cnt(dec)+1);

        cnt(dec) = cnt(dec)+1;
        cntNot(decNot) = cntNot(decNot)+1;

        if t <= K
            att = 0; % non esguo nessun attacco
        else
            att = -ind(dec) *nanmax(0,avgU(dec) - avgU(K) ...
                      + beta(cnt(dec)) + beta(cnt(K)));
        end
        c(t,r) = abs(att); % norma 1
        rewa(t,dec,r) = rw + att;
        %avg(dec) = (avg(dec)*(cnt(dec)-1) + rw  + att)/cnt(dec); 
        Target(t,r) = dec;

        % se non ci fose stato l'attacco: 
        rewaNot(t,decNot,r) = rwn;
        TargetNot(t,r) = decNot; 
    end
end

Target = Target == target;
TargetNot = TargetNot == target;

figure
subplot(2,1,1)
semilogx([1:T],cumsum(nanmean(c,2)));
grid on
title("Costo cumulato UBC")
ylabel("Costo")
xlabel("Time step")
legend(["UBC"],'Location','Best')

subplot(2,1,2)
plot([1:T],cumsum(nanmean(Target,2)));
hold on
plot([1:T],cumsum(nanmean(TargetNot,2)),'g');
grid on 
grid minor
title("#Pulled arm a^*")
ylabel("Pulled arm")
xlabel("Time step")
legend(["ACE Attack","No Attack"],'Location','Best')

% savefig("online_UBC_normale")
% print("online_UBC_normale",'-dpng')


%% THOMSOM SAMPLING (Normal) ONLINE 

rewa = NaN(T,K,R);
Target = NaN(T,R);
TargetNot = NaN(T,K);
rewaNot = NaN(T,K,R); % reward

for r = 1:R
   cnt = zeros(1,K);
   cntNot = zeros(1,K);
   avgU = zeros(1,K);
    for t = 1:T
        if t<=K
            dec = t;
            decNot = dec; 
        else 
            rd = randn;
            avg = nanmean(rewa(:,:,r)); % mean branch (bias) 
            avg(isnan(avg)) = 0;
            [m ,dec] = nanmax(avg/(sigma^2) + rd*sqrt(sigma^2./cnt));
            
            avg = nanmean(rewaNot(:,:,r)); % mean branch (bias) 
            avg(isnan(avg)) = 0;
            [m ,decNot] = nanmax(avg/(sigma^2) + rd*sqrt(sigma^2./cntNot));
            
        end
  
        rd = randn; 
        rw = mu(dec) + rd*sigma;
        rwn = mu(decNot) + rd*sigma;

        avgU(dec) = (avgU(dec) * (cnt(dec)) + rw) / (cnt(dec) + 1);
        
        cnt(dec) = cnt(dec)+1;
        cntNot(decNot) = cntNot(decNot)+1;
              
        if t <= K
            att = 0; % non esguo nessun attocco
        else
            att = -ind(dec) *nanmax(0,avgU(dec) - avgU(K) ...
                      + beta(cnt(dec)) + beta(cnt(K)));
        end
        c(t,r) = abs(att); % norma 1
        rewa(t,dec,r) = rw + att;
        Target(t,r) = dec;

        rewaNot(t,decNot,r) = rwn;
        TargetNot(t,r) = decNot;
          
    end
end

Target = Target == target;
TargetNot = TargetNot == target;

figure
subplot(2,1,1)
semilogx([1:T],cumsum(nanmean(c,2)));
grid on
title("Costo cumulato Thompson Sampling")
ylabel("Costo")
xlabel("Time step")
legend(["Thompson Sampling"],'Location','Best')

subplot(2,1,2)
plot([1:T],cumsum(nanmean(Target,2)));
hold on
plot([1:T],cumsum(nanmean(TargetNot,2)),'g');
grid on 
grid minor
title("#Pulled arm a^*")
ylabel("Pulled arm")
xlabel("Time step")
legend(["ACE Attack","No Attack"],'Location','Best')

% savefig("online_thompson_normale")
% print("online_thompson_normale",'-dpng')

