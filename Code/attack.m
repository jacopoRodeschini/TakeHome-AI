function [avg,cost] = attack(cnt,avg,average,opt)
    
    T = opt.M;
    K = size(cnt,2); 
    H = eye(T);
    
    % e-greedy 
    A = zeros(K-1,T);
    off = 0;

    if strcmp( opt.type , 'e-greedy')
        for i = 1:K-1
            A(i,off+1:off+cnt(i)) = 1/cnt(i)*ones(1,cnt(i));
            A(i,T-cnt(K)+1:T) = -1/cnt(K)*ones(1,cnt(K));
            off = off + cnt(i);
        end
        
        
        b = ones(K-1,1)*avg(K) - avg(1:K-1)' - opt.xi;
        
        %b = ones(K-1,1)*avg(K) - avg(1:K-1)' - sqrt(opt.gamma * log(opt.T*opt.K) ./ opt.tot)  
        
        C = zeros(K,T);
        C(1:K-1,1:T-cnt(K)) = A(:,1:T-cnt(K));
        C(K,T-cnt(K)+1:T) = 1/cnt(K)* ones(1,cnt(K)); 
        
    elseif strcmp(opt.type,'Base')

        rho =  opt.rho;
        T = opt.M; 
        off = 0;
        for i = 1:K-1
            A(i,off+1:off+cnt(i)) = 1/cnt(i)*(1-rho(i))*ones(1,cnt(i));
            A(i,T-cnt(K)+1:T) = -1/cnt(K)*(1-rho(K))*ones(1,cnt(K));
            off = off + cnt(i);
        end
        
        b = ones(1,K-1).*(average(K)*rho(K) + avg(K)*(1-rho(K)))+...
        sqrt(opt.gamma * log(opt.T*opt.K) ./ opt.tot)*ones(1,K-1)...
        - average(1:K-1).*rho(1:K-1) - avg(1:K-1).*(1-rho(1:K-1));
    
        C = zeros(K,T);
        off = 0; 
        for i = 1:K
            C(i,off+1:off+cnt(i)) = 1/cnt(i)*ones(1,cnt(i));
            off = off + cnt(i);
        end
        
        %C(1:K-1,1:T-cnt(K)) = A(:,1:T-cnt(K))./(1 - rho(1:K-1))';
        %C(K,T-cnt(K)+1:T) = 1/cnt(K)* ones(1,cnt(K)); 
    end
    
    eps = quadprog(H,[],A,b,[],[],[],[],[],opt.opt);

    avg = avg + (C*eps)';
   
    cost = eps'*eps;
end

