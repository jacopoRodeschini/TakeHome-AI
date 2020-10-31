function [cost,ch_next] = meanBasedAttackUCB(ch,succ,opts)

sigma = opts.sigma; % varianza dell'errore
opt = opts.opt;     % opzioni per programmazione linerare
ski = opts.ski;     % variabile surplus programmazione linerare

K = size(ch,2);
R = size(ch,3);
T = size(ch,1);
ch_next = NaN(R,1);
cost = NaN(R,1);
err = @(t,x)sqrt(log(t)./x);
delta = opts.delta;
    for j = 1:R    

        % effettuiamo l'attacco    
        % prendiamo tutto il batch size T e manipoliamo i raward affinche la
        % scelta ottima sia il batch 5 (prob = 0).
        % verifichiamo infine che al T+1 il braccio scelto sia effettivamente
        % quello desiderato (sub-ottimo).

        % info che servono all'manipolatore
        cnt = nansum(ch(:,:,j));
        cnt(isnan(cnt)) = 0; 
        avg = nanmean(succ(:,:,j)); % media stimata
        % inizializzazione problema di ottimizzazione (linear programming)

        H = eye(T);
        A = zeros(K-1,T); % vincoli sul problema di ottimizzazione
        of = 0;
        % att: K è il nostro traget

        % matrice dei coefficienti A
        % attenzione a come sono ordinati i coeff. all'interno della matrice i
        % risultati non cambiano ma cambia l'interpretazione 
        for i = 1:(K-1)
            A(i,of+1:of+cnt(i)) =   ones(1,cnt(i))./cnt(i); 
            A(i,T-cnt(K)+1:T) =   -ones(1,cnt(K))./cnt(K); 
            of = of + cnt(i);
        end

        % vettore dei termini noti 
        b = ones(K-1,1)*avg(K);
        b = b - avg(1:end-1)' - ski - 3*sigma*err(T+1,(cnt(1:end-1))') ... 
             + 3*sigma*err(T+1,(cnt(K)));

        % risolutore 
        % x: soluzione ottima ovvero overmo tutti i "rumori" dallo step 1 allo
        % step T, ordinati come la matrice A da applicare ai reward ottenuti 
        x = quadprog(H,[],A,b,[],[],[],[],[],opt); 


        % verifichiamo la soluzione
        % ovvero che al tempo T+1 il braccio scelto sia appunto il braccio K,
        % per far cio ricalcoliamo le medie dei vari bracci come la media
        % precednente + il contributo del rumore additivo

        C = zeros(K,T); % matrice dei coeff. 
        C(1:end-1,1:T-cnt(K)) = A(:,1:T-cnt(K));
        C(end,T-cnt(K)+1:end) = ones(1,cnt(K))/cnt(K);

        % calocolo nuove scelta tramite l'update delle medie
        [m, t] = max(avg + (C*x)' + 3 *sigma * err(T,cnt));

        % scelta al passo al T + 1
        ch_next(j) = t;

        % ratio
        cost(j) = x' * x ;   

    end

end

