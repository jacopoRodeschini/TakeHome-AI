function [cnt,avg] = computeBatch(T,avgOld,cntOld,opt)

    mu = opt.mu; % media vera rw del braccio  
    sigma = opt.sigma; % varianza errore
    eps = opt.eps;
    K = size(avgOld,2);
    cnt = zeros(1,K);
    avg = zeros(1,K);
    for t = 1:T
        if t<=K
            dec = t;
        else
            if rand < eps
                dec = randi(K);
            else 
                [m,dec] = nanmax(avgOld);
            end
        end
        % simulazione ambiente esterno 
        
        rw = mu(dec) + randn*sigma;
        avg(dec)  = (avg(dec)* cnt(dec) + rw)/ (cnt(dec) + 1);
        cnt(dec) = cnt(dec) + 1; 
    end
    p = cntOld./(cnt + cntOld);
    avg =  avgOld.*p + avg.*(1 - p);
end

