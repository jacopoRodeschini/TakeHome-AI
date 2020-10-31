function [cost,ch_next] = meanBasedAttackThompson(ch,succ,opts)

K = size(succ,2);
T = size(succ,1);
R = size(succ,3);

sigma = opts.sigma; % varianza dell'errore
opt = opts.opt;     % opzioni per programmazione linerare
ski = opts.ski;     % variabile surplus programmazione linerare
delta = opts.delta; % p = 1 - delta 

ch_next = NaN(R,1);
cost = NaN(R,1);


for j = 1:R    
     
    % info che servono al manipolatore
    cnt = nansum(ch(:,:,j)); % frequenza delle scelte
    avg = nanmean(succ(:,:,j)); % media stimata
    % inizializzazione problema di ottimizzazione (linear programming)
    
    H = eye(T);
    
    % vincoli sul problema di ottimizzazione,
    % poichè ho scelto un sigma che soddisfa la disequazione, posso evitare
    % di inserire il secondo vincolo, poichè risulta che il primo è molto
    % piu stringente 
    A = zeros(K-1,T); 
    of = 0; 
    % attenzione: il branch K è il nostro traget
    
    % matrice dei coefficienti A
    % attenzione a come sono ordinati i coeff. all'interno della matrice, i
    % risultati non cambiano ma cambia l'interpretazione 
    for i = 1:(K-1)
        A(i,of+1:of+cnt(i)) =   ones(1,cnt(i))./cnt(i); 
        A(i,T-cnt(K)+1:T) =   -ones(1,cnt(K))./cnt(K); 
        of = of + cnt(i);
    end
    % va agg. un vincolo sulla riga K
    
    % vettore dei termini noti  
    b = ones(K-1,1)*avg(K);
    b = b - avg(1:end-1)' + ...
        norminv(delta/(K-1))*sigma^3*sqrt(1./cnt(1:K-1) + 1/cnt(K))';
    
    % risolutore 
    % x: soluzione ottima ovvero overmo tutti i "rumori" dallo step 1 allo
    % step T, ordinati come la matrice A da applicare ai reward ottenuti 
    
    x = quadprog(H,[],A,b,[],[],[],[],[],opt); % SOLUZIONE AMMISSIBILE
    
    % verifichiamo la soluzione
    % ovvero che al tempo T+1 il braccio scelto sia appunto il braccio K,
    % per far cio ricalcoliamo 
    
    C = zeros(K,T); % matrice dei coeff. 
    C(1:end-1,1:T-cnt(K)) = A(:,1:T-cnt(K));
    C(end,T-cnt(K)+1:end) = ones(1,cnt(K))/cnt(K);
    
    
    % costo tot
    c = 0;
    for i = 1:K-1
        c = c + normcdf(((avg(i) + C(i,:)*x) - (avg(i) + C(K,:)*x))...
            / (sigma^3 * sqrt(1/cnt(i) + 1/cnt(K))) );
    end
    c = c - delta;
    if c <= 0 
        % calocolo nuove scelta tramite l'update delle medie
        [m, t] = max(avg + (C*x)');
        % scelta al passo al T + 1
        ch_next(j) = t;

        % ratio
        cost(j) = x' * x;
    end
end


end

