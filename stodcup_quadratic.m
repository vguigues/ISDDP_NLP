
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% T: number of stages
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% n: size of state vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% M: number of scenarios
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% iter_stodcup: number of stodcup iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% nb_iter _max: maximal number of iterations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% xis{1,t}(k,j): scenario k component j for stage t.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Us{1,t}(k): value of U_t for scenario k
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Psis{1,t}(k): value of Psi_t for scenario k
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tol: tolerance for the stopping criterion.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Outputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% upper_bounds _sddp(k) is the upper bound for SDDP iteration k.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lower_bounds _sddp(k) is the lower bound for SDDP iteration k.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% time_sddp(k) is the CPU time needed to solve iteration k of the problem
% with SDDP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% upper_bounds _stodcup(k) is the upper bound for StoDCUP iteration k.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% lower_bounds _stodcup(k) is the lower bound for StoDCUP iteration k.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% time_stodcup(k) is the CPU time needed to solve iteration k of the problem
% with StoDCUP
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [lower_bounds_sddp,upper_bounds_sddp,time_sddp]=stodcup_quadratic(T,n,M,iter_stodcup,nb_iter_max,xis,Us,Psis,tol,x0,Minit,probabilities)

lower_bounds_sddp=[];
upper_bounds_sddp=[];
time_sddp=[];
lower_bounds_stodcup=[];
upper_bounds_stodcup=[];
time_stodcup=[];

subia=cell(1,T-1);
subja=cell(1,T-1);
valija=cell(1,T-1);

subistodcup=cell(1,T);
subjstodcup=cell(1,T);
valijstodcup=cell(1,T);

subistodcupg=cell(1,T);
 subjstodcupg=cell(1,T);
valijstodcupg=cell(1,T);

bs=cell(1,T);
cs=cell(1,T);
hs=cell(1,T);
thetas=cell(1,T-1);

for t=1:(T-1)
    subia{1,t}=[subia{1,t},1];
    subja{1,t}=[subja{1,t},2];
    valija{1,t}=[valija{1,t},1];
    thetas{1,t}= -(10^(10));
end

counters=zeros(T,M);
for t=1:T
    subistodcup{1,t}=cell(1,M);
    subjstodcup{1,t}=cell(1,M);
    valijstodcup{1,t}=cell(1,M);
    subistodcupg{1,t}=cell(1,M);
    subjstodcupg{1,t}=cell(1,M);
    valijstodcupg{1,t}=cell(1,M);
    bs{1,t}=cell(1,M);
    cs{1,t}=cell(1,M);
    hs{1,t}=cell(1,M);
    if (t==1)
        Maux=1;
    else
        Maux=M;
    end
    for j=1:Maux
        xi=xis{1,t}(j,:)';
        ur=Us{1,t}(j);
        psir=Psis{1,t}(j);
        for i=1:Minit
            x1=-100*ones(n,1)+200*rand(n,1);
            if (t>=2)
                x2=-100*ones(n,1)+200*rand(n,1);
            else
                x2=x0;
            end
            v=x1-x2;
            f1=v'*xi*xi'*v+x1'*xi+1;
            f2=x1'*xi*xi'*x1+x1'*ones(n,1)+ur;
            [vf,indf]=max([f1,f2]);
            g1=4*((x1-ones(n,1))'*(x1-ones(n,1)));
            g2=x1'*xi*xi'*x1+x1'*xi+1;
            [vg,indg]=max([g1,g2]);
            counters(t,j)=counters(t,j)+1;
            if (indf==1)
                if (t==1)
                    subistodcup{1,t}{1,j}=[subistodcup{1,t}{1,j},counters(t,j)*ones(1,n+1)];
                    subjstodcup{1,t}{1,j}=[subjstodcup{1,t}{1,j},1,[3:1:n+2]];
                    valijstodcup{1,t}{1,j}=[valijstodcup{1,t}{1,j},1,-2*(x1-x2)'*xi*xi'-xi'];
                    bs{1,t}{1,j}=[bs{1,t}{1,j};zeros(1,n)];
                    cs{1,t}{1,j}=[cs{1,t}{1,j};1-(x1-x2)'*xi*xi'*(x1+x2)];
                else
                    subistodcup{1,t}{1,j}=[subistodcup{1,t}{1,j},counters(t,j)*ones(1,n+1)];
                    subjstodcup{1,t}{1,j}=[subjstodcup{1,t}{1,j},1,[3:1:n+2]];
                    valijstodcup{1,t}{1,j}=[valijstodcup{1,t}{1,j},1,-2*(x1-x2)'*xi*xi'-xi'];
                    bs{1,t}{1,j}=[bs{1,t}{1,j};2*(x2-x1)'*xi*xi'];
                    cs{1,t}{1,j}=[cs{1,t}{1,j};1-(x1-x2)'*xi*xi'*(x1-x2)];
                end
            else
                subistodcup{1,t}{1,j}=[subistodcup{1,t}{1,j},counters(t,j)*ones(1,n+1)];
                subjstodcup{1,t}{1,j}=[subjstodcup{1,t}{1,j},1,[3:1:n+2]];
                valijstodcup{1,t}{1,j}=[valijstodcup{1,t}{1,j},1,-2*x1'*xi*xi'-ones(1,n)];
                cs{1,t}{1,j}=[cs{1,t}{1,j};ur-x1'*xi*xi'*x1];
                bs{1,t}{1,j}=[bs{1,t}{1,j};zeros(1,n)];
            end
            if (indg==1)
                subistodcupg{1,t}{1,j}=[subistodcupg{1,t}{1,j},counters(t,j)*ones(1,n)];
                subjstodcupg{1,t}{1,j}=[subjstodcupg{1,t}{1,j},[3:1:n+2]];
                valijstodcupg{1,t}{1,j}=[valijstodcupg{1,t}{1,j},8*(x1'-ones(1,n))];
                hs{1,t}{1,j}=[hs{1,t}{1,j};psir+4*(x1-ones(n,1))'*(x1+ones(n,1))];
            else
                subistodcupg{1,t}{1,j}=[subistodcupg{1,t}{1,j},counters(t,j)*ones(1,n)];
                subjstodcupg{1,t}{1,j}=[subjstodcupg{1,t}{1,j},[3:1:n+2]];
                valijstodcupg{1,t}{1,j}=[valijstodcupg{1,t}{1,j},xi'+2*x1'*xi*xi'];
                hs{1,t}{1,j}=[hs{1,t}{1,j};psir+x1'*xi*xi'*x1-1];
            end
        end
    end
end

Cum_Probas=cell(1,T-1);
for t=1:T-1
    Cum_Probas{1,t}=[0,cumsum(probabilities{1,t})];
end

End_Algo=1;
iter=1;
Costs=[];

while End_Algo
    %iter
    %Forward pass
    total_cost=0;
    taux1=0;
    trial_states=cell(1,T-1);
    if (iter<=iter_stodcup)
        tic
        clear prob; 
        prob.c=[1,1,zeros(1,n)];
        prob.blx=[-inf;-10^(10);-100*ones(n,1)];
        prob.bux=[inf;inf;100*ones(n,1)];
        prob.buc=[inf*ones(counters(1,1),1);hs{1,1}{1,1};inf*ones(iter,1)];
        prob.blc=[cs{1,1}{1,1};-inf*ones(counters(1,1),1);thetas{1,1}];
        prob.a=sparse([subia{1,1}+2*counters(1,1),subistodcup{1,1}{1,1},subistodcupg{1,1}{1,1}+counters(1,1)],[subja{1,1},subjstodcup{1,1}{1,1},subjstodcupg{1,1}{1,1}],[valija{1,1},valijstodcup{1,1}{1,1},valijstodcupg{1,1}{1,1}],2*counters(1,1)+iter,n+2);
        [~,res]=mosekopt('minimize echo(0)',prob);
        sol=res.sol.itr.xx;
        solsta=strcat('MSK_SOL_STA_', res.sol.itr.solsta);
        if (strcmp(solsta,'MSK_SOL_STA_PRIMAL_INFEASIBLE_CER')==1)
            disp('Unfeasible primal problem');
            trial_states{1,1}=sol(3:n+2);
            %pause
        elseif (strcmp(solsta,'MSK_SOL_STA_DUAL_INFEASIBLE_CER')==1)
            disp('Primal infinite optimal value');
            trial_states{1,1}=sol(3:n+2);
            %pause
        else
            trial_states{1,1}=sol(3:n+2);
        end
        zinf=sol(1)+sol(2);
        lower_bounds_sddp=[lower_bounds_sddp;zinf];
        total_cost=total_cost+sol(1);
        taux1=taux1+toc;
        
        xi=xis{1,1}(1,:)';
        ur=Us{1,1}(1);
        psir=Psis{1,1}(1);
        x1=sol(3:n+2);
        x2=x0;
        v=x1-x2;
        f1=v'*xi*xi'*v+x1'*xi+1;
        f2=x1'*xi*xi'*x1+x1'*ones(n,1)+ur;
        [vf,indf]=max([f1,f2]);
        g1=4*((x1-ones(n,1))'*(x1-ones(n,1)));
        g2=x1'*xi*xi'*x1+x1'*xi+1;
        [vg,indg]=max([g1,g2]);
        counters(1,1)=counters(1,1)+1;
        if (indf==1)
            subistodcup{1,1}{1,1}=[subistodcup{1,1}{1,1},counters(1,1)*ones(1,n+1)];
            subjstodcup{1,1}{1,1}=[subjstodcup{1,1}{1,1},1,[3:1:n+2]];
            valijstodcup{1,1}{1,1}=[valijstodcup{1,1}{1,1},1,-2*(x1-x2)'*xi*xi'-xi'];
            cs{1,1}{1,1}=[cs{1,1}{1,1};1-(x1-x2)'*xi*xi'*(x1+x2)];
        else
            subistodcup{1,1}{1,1}=[subistodcup{1,1}{1,1},counters(1,1)*ones(1,n+1)];
            subjstodcup{1,1}{1,1}=[subjstodcup{1,1}{1,1},1,[3:1:n+2]];
            valijstodcup{1,1}{1,1}=[valijstodcup{1,1}{1,1},1,-2*x1'*xi*xi'-ones(1,n)];
            cs{1,1}{1,1}=[cs{1,1}{1,1};ur-x1'*xi*xi'*x1];
        end
        if (indg==1)
            subistodcupg{1,1}{1,1}=[subistodcupg{1,1}{1,1},counters(1,1)*ones(1,n)];
            subjstodcupg{1,1}{1,1}=[subjstodcupg{1,1}{1,1},[3:1:n+2]];
            valijstodcupg{1,1}{1,1}=[valijstodcupg{1,1}{1,1},8*(x1'-ones(1,n))];
            hs{1,1}{1,1}=[hs{1,1}{1,1};psir+4*(x1-ones(n,1))'*(x1+ones(n,1))];
        else
            subistodcupg{1,1}{1,1}=[subistodcupg{1,1}{1,1},counters(1,1)*ones(1,n)];
            subjstodcupg{1,1}{1,1}=[subjstodcupg{1,1}{1,1},[3:1:n+2]];
            valijstodcupg{1,1}{1,1}=[valijstodcupg{1,1}{1,1},xi'+2*x1'*xi*xi'];
            hs{1,1}{1,1}=[hs{1,1}{1,1};psir+x1'*xi*xi'*x1-1];
        end
        
        for t=2:T-1
            tic
            clear prob;
            Alea_Uniform=rand;
            [~,Index] = histc(Alea_Uniform,Cum_Probas{1,t-1});
            if (Alea_Uniform==1)
                Index=M;
            end
            clear prob;
            prob.c=[1,1,zeros(1,n)];
            prob.blx=[-inf;-10^(10);-100*ones(n,1)];
            prob.bux=[inf;inf;100*ones(n,1)];
            prob.buc=[inf*ones(counters(t,Index),1);hs{1,t}{1,Index};inf*ones(iter,1)];
            prob.blc=[bs{1,t}{1,Index}*trial_states{1,t-1}+cs{1,t}{1,Index};-inf*ones(counters(t,Index),1);thetas{1,t}];
            prob.a=sparse([subia{1,t}+2*counters(t,Index),subistodcup{1,t}{1,Index},subistodcupg{1,t}{1,Index}+counters(t,Index)],[subja{1,t},subjstodcup{1,t}{1,Index},subjstodcupg{1,t}{1,Index}],[valija{1,t},valijstodcup{1,t}{1,Index},valijstodcupg{1,t}{1,Index}],2*counters(t,Index)+iter,n+2);
            [~,res]=mosekopt('minimize echo(0)',prob);
            sol=res.sol.itr.xx;
            solsta=strcat('MSK_SOL_STA_', res.sol.itr.solsta);
            if (strcmp(solsta,'MSK_SOL_STA_PRIMAL_INFEASIBLE_CER')==1)
                disp('Unfeasible primal problem');
                trial_states{1,1}=sol(3:n+2);
                total_cost=total_cost+sol(1);
                %pause
            elseif (strcmp(solsta,'MSK_SOL_STA_DUAL_INFEASIBLE_CER')==1)
                disp('Primal infinite optimal value');
                trial_states{1,1}=sol(3:n+2);
                total_cost=total_cost+sol(1);
                %pause
            else
                trial_states{1,t}=sol(3:n+2);
                total_cost=total_cost+sol(1);
            end
            taux1=taux1+toc;
            xi=xis{1,t}(Index,:)';
            ur=Us{1,t}(Index);
            psir=Psis{1,t}(Index);
            x1=sol(3:n+2);
            x2=trial_states{1,t-1};
            v=x1-x2;
            f1=v'*xi*xi'*v+x1'*xi+1;
            f2=x1'*xi*xi'*x1+x1'*ones(n,1)+ur;
            [vf,indf]=max([f1,f2]);
            g1=4*((x1-ones(n,1))'*(x1-ones(n,1)));
            g2=x1'*xi*xi'*x1+x1'*xi+1;
            [vg,indg]=max([g1,g2]);
            counters(t,Index)=counters(t,Index)+1;
            if (indf==1)
                subistodcup{1,t}{1,Index}=[subistodcup{1,t}{1,Index},counters(t,Index)*ones(1,n+1)];
                subjstodcup{1,t}{1,Index}=[subjstodcup{1,t}{1,Index},1,[3:1:n+2]];
                valijstodcup{1,t}{1,Index}=[valijstodcup{1,t}{1,Index},1,-2*(x1-x2)'*xi*xi'-xi'];
                bs{1,t}{1,Index}=[bs{1,t}{1,Index};2*(x2-x1)'*xi*xi'];
                cs{1,t}{1,Index}=[cs{1,t}{1,Index};1-(x1-x2)'*xi*xi'*(x1-x2)];
            else
                subistodcup{1,t}{1,Index}=[subistodcup{1,t}{1,Index},counters(t,Index)*ones(1,n+1)];
                subjstodcup{1,t}{1,Index}=[subjstodcup{1,t}{1,Index},1,[3:1:n+2]];
                valijstodcup{1,t}{1,Index}=[valijstodcup{1,t}{1,Index},1,-2*x1'*xi*xi'-ones(1,n)];
                cs{1,t}{1,Index}=[cs{1,t}{1,Index};ur-x1'*xi*xi'*x1];
                bs{1,t}{1,Index}=[bs{1,t}{1,Index};zeros(1,n)];
            end
            if (indg==1)
                subistodcupg{1,t}{1,Index}=[subistodcupg{1,t}{1,Index},counters(t,Index)*ones(1,n)];
                subjstodcupg{1,t}{1,Index}=[subjstodcupg{1,t}{1,Index},[3:1:n+2]];
                valijstodcupg{1,t}{1,Index}=[valijstodcupg{1,t}{1,Index},8*(x1'-ones(1,n))];
                hs{1,t}{1,Index}=[hs{1,t}{1,Index};psir+4*(x1-ones(n,1))'*(x1+ones(n,1))];
            else
                subistodcupg{1,t}{1,Index}=[subistodcupg{1,t}{1,Index},counters(t,Index)*ones(1,n)];
                subjstodcupg{1,t}{1,Index}=[subjstodcupg{1,t}{1,Index},[3:1:n+2]];
                valijstodcupg{1,t}{1,Index}=[valijstodcupg{1,t}{1,Index},xi'+2*x1'*xi*xi'];
                hs{1,t}{1,Index}=[hs{1,t}{1,Index};psir+x1'*xi*xi'*x1-1];
            end
        end
        
        tic
        clear prob;
        Alea_Uniform=rand;
        [~,Index] = histc(Alea_Uniform,Cum_Probas{1,T-1});
        if (Alea_Uniform==1)
            Index=M;
        end
        clear prob;
        prob.c=[1,zeros(1,n+1)];
        prob.blx=[-inf;0;-100*ones(n,1)];
        prob.bux=[inf;0;100*ones(n,1)];
        prob.buc=[inf*ones(counters(T,Index),1);hs{1,T}{1,Index}];
        prob.blc=[bs{1,T}{1,Index}*trial_states{1,T-1}+cs{1,T}{1,Index};-inf*ones(counters(T,Index),1)];
        prob.a=sparse([subistodcup{1,T}{1,Index},subistodcupg{1,T}{1,Index}+counters(T,Index)],[subjstodcup{1,T}{1,Index},subjstodcupg{1,T}{1,Index}],[valijstodcup{1,T}{1,Index},valijstodcupg{1,T}{1,Index}],2*counters(T,Index),n+2);
        [~,res]=mosekopt('minimize echo(0)',prob);
        sol=res.sol.itr.xx;
        solsta=strcat('MSK_SOL_STA_', res.sol.itr.solsta);
        if (strcmp(solsta,'MSK_SOL_STA_PRIMAL_INFEASIBLE_CER')==1)
            disp('Unfeasible primal problem');
            total_cost=total_cost+sol(1);
            %pause
        elseif (strcmp(solsta,'MSK_SOL_STA_DUAL_INFEASIBLE_CER')==1)
            disp('Primal infinite optimal value');
            total_cost=total_cost+sol(1);
            %pause
        else
            total_cost=total_cost+sol(1);
        end
        taux1=taux1+toc;
        
        xi=xis{1,T}(Index,:)';
        ur=Us{1,T}(Index);
        psir=Psis{1,T}(Index);
        x1=sol(3:n+2);
        x2=trial_states{1,T-1};
        v=x1-x2;
        f1=v'*xi*xi'*v+x1'*xi+1;
        f2=x1'*xi*xi'*x1+x1'*ones(n,1)+ur;
        [vf,indf]=max([f1,f2]);
        g1=4*((x1-ones(n,1))'*(x1-ones(n,1)));
        g2=x1'*xi*xi'*x1+x1'*xi+1;
        [vg,indg]=max([g1,g2]);
        counters(T,Index)=counters(T,Index)+1;
        if (indf==1)
            subistodcup{1,T}{1,Index}=[subistodcup{1,T}{1,Index},counters(T,Index)*ones(1,n+1)];
            subjstodcup{1,T}{1,Index}=[subjstodcup{1,T}{1,Index},1,[3:1:n+2]];
            valijstodcup{1,T}{1,Index}=[valijstodcup{1,T}{1,Index},1,-2*(x1-x2)'*xi*xi'-xi'];
            bs{1,T}{1,Index}=[bs{1,T}{1,Index};2*(x2-x1)'*xi*xi'];
            cs{1,T}{1,Index}=[cs{1,T}{1,Index};1-(x1-x2)'*xi*xi'*(x1-x2)];
        else
            subistodcup{1,T}{1,Index}=[subistodcup{1,T}{1,Index},counters(T,Index)*ones(1,n+1)];
            subjstodcup{1,T}{1,Index}=[subjstodcup{1,T}{1,Index},1,[3:1:n+2]];
            valijstodcup{1,T}{1,Index}=[valijstodcup{1,T}{1,Index},1,-2*x1'*xi*xi'-ones(1,n)];
            cs{1,T}{1,Index}=[cs{1,T}{1,Index};ur-x1'*xi*xi'*x1];
            bs{1,T}{1,Index}=[bs{1,T}{1,Index};zeros(1,n)];
        end
        if (indg==1)
            subistodcupg{1,T}{1,Index}=[subistodcupg{1,T}{1,Index},counters(T,Index)*ones(1,n)];
            subjstodcupg{1,T}{1,Index}=[subjstodcupg{1,T}{1,Index},[3:1:n+2]];
            valijstodcupg{1,T}{1,Index}=[valijstodcupg{1,T}{1,Index},8*(x1'-ones(1,n))];
            hs{1,T}{1,Index}=[hs{1,T}{1,Index};psir+4*(x1-ones(n,1))'*(x1+ones(n,1))];
        else
            subistodcupg{1,T}{1,Index}=[subistodcupg{1,T}{1,Index},counters(T,Index)*ones(1,n)];
            subjstodcupg{1,T}{1,Index}=[subjstodcupg{1,T}{1,Index},[3:1:n+2]];
            valijstodcupg{1,T}{1,Index}=[valijstodcupg{1,T}{1,Index},xi'+2*x1'*xi*xi'];
            hs{1,T}{1,Index}=[hs{1,T}{1,Index};psir+x1'*xi*xi'*x1-1];
        end
    else
        total_cost=0;
        trial_states=cell(1,T-1);
        for t=1:T
            if (t==1)
                Index=1;
            else
                Alea_Uniform=rand;
                [~,Index] = histc(Alea_Uniform,Cum_Probas{1,t-1});
                if (Alea_Uniform==1)
                    Index=M;
                end
            end
            
            tic
            xi=xis{1,t}(Index,:)';
            ur=Us{1,t}(Index);
            psir=Psis{1,t}(Index);
            clear prob;
            prob.qcsubk=[];
            prob.qcsubi=[];
            prob.qcsubj=[];
            prob.qcval=[];
            subi=[];
            subj=[];
            valij=[];
            prob.c=[1,1,zeros(1,n+1)];
            if (t==T)
                prob.blx=[-inf;0;-100*ones(n,1);-inf];
                prob.bux=[inf;0;100*ones(n,1);inf];
            else
                prob.blx=[-inf;-10^(10);-100*ones(n,1);-inf];
                prob.bux=[inf;inf;100*ones(n,1);inf];
            end
            if (t==1)
                prob.buc=[-1-x0'*xi*xi'*x0;-ur;-4*n;-1;psir;inf*ones(iter,1)];
            else
                if (t<T)
                    prob.buc=[-1-trial_states{1,t-1}'*xi*xi'*trial_states{1,t-1};-ur;-4*n;-1;psir;inf*ones(iter,1)];
                else
                    prob.buc=[-1-trial_states{1,t-1}'*xi*xi'*trial_states{1,t-1};-ur;-4*n;-1;psir];
                end     
            end
            if (t<T)
                prob.blc=[-inf;-inf;-inf;-inf;-inf;thetas{1,t}];
            else
                prob.blc=[-inf;-inf;-inf;-inf;-inf];
            end       
            %Constraint 1
            subi=[subi,ones(1,n+1)];
            subj=[subj,1,[3:1:n+2]];
            if (t==1)
                valij=[valij,-1,xi'-2*x0'*xi*xi'];
            else
                valij=[valij,-1,xi'-2*trial_states{1,t-1}'*xi*xi'];                
            end
            for i=1:n
                for j=1:i
                    prob.qcsubk=[prob.qcsubk,1];
                    prob.qcsubi=[prob.qcsubi,2+i];
                    prob.qcsubj=[prob.qcsubj,2+j];
                    prob.qcval=[prob.qcval,2*xi(i)*xi(j)];
                end
            end
            
            %Constraint 2
            subi=[subi,2*ones(1,n+1)];
            subj=[subj,1,[3:1:n+2]];
            valij=[valij,-1,ones(1,n)];
            for i=1:n
                for j=1:i
                    prob.qcsubk=[prob.qcsubk,2];
                    prob.qcsubi=[prob.qcsubi,2+i];
                    prob.qcsubj=[prob.qcsubj,2+j];
                    prob.qcval=[prob.qcval,2*xi(i)*xi(j)];
                end
            end
            
            %Constraint 3
            subi=[subi,3*ones(1,n+1)];
            subj=[subj,n+3,[3:1:n+2]];
            valij=[valij,-1,-8*ones(1,n)];
            for i=1:n
                prob.qcsubk=[prob.qcsubk,3];
                prob.qcsubi=[prob.qcsubi,2+i];
                prob.qcsubj=[prob.qcsubj,2+i];
                prob.qcval=[prob.qcval,8];
            end
            
            %Constraint 4
            subi=[subi,4*ones(1,n+1)];
            subj=[subj,n+3,[3:1:n+2]];
            valij=[valij,-1,xi'];
            for i=1:n
                for j=1:i
                    prob.qcsubk=[prob.qcsubk,4];
                    prob.qcsubi=[prob.qcsubi,2+i];
                    prob.qcsubj=[prob.qcsubj,2+j];
                    prob.qcval=[prob.qcval,2*xi(i)*xi(j)];
                end
            end
            
            %Constraint 5
            subi=[subi,5];
            subj=[subj,n+3];
            valij=[valij,1];
            
            if (t<T)
                subi=[subi,subia{1,t}+5];
                subj=[subj,subja{1,t}];
                valij=[valij,valija{1,t}];
            end
            
            if (t==T)
                prob.a=sparse(subi,subj,valij,5,n+3);
            else
                prob.a=sparse(subi,subj,valij,5+iter,n+3);
            end
            
            [~,res]=mosekopt('minimize echo(0)',prob);
            sol=res.sol.itr.xx;
            solsta=strcat('MSK_SOL_STA_', res.sol.itr.solsta);
            if (strcmp(solsta,'MSK_SOL_STA_PRIMAL_INFEASIBLE_CER')==1)
                disp('Unfeasible primal problem');
                if (t<T)
                    trial_states{1,t}=sol(3:n+2);
                end
                %pause
            elseif (strcmp(solsta,'MSK_SOL_STA_DUAL_INFEASIBLE_CER')==1)
                disp('Primal infinite optimal value');
                if (t<T)
                    trial_states{1,t}=sol(3:n+2);
                end
                %pause
            else
                if (t<T)
                    trial_states{1,t}=sol(3:n+2);
                end
            end
            if (t==1)
                zinf=sol(1)+sol(2);
                lower_bounds_sddp=[lower_bounds_sddp;zinf];
            end
            total_cost=total_cost+sol(1);
            taux1=taux1+toc;
        end
    end
    
    Costs=[Costs;total_cost];
    zsup=mean(Costs)+1.96*sqrt(var(Costs))/sqrt(iter);
    upper_bounds_sddp=[upper_bounds_sddp;zsup];
    
    % Backward pass
    
    if (t<=iter_stodcup)
    
    for t=T:-1:2
        if (t==T)
            intercept=0;
            slope=zeros(n,1);
            for j=1:M
                tic
                clear prob;
                prob.c=[1,zeros(1,n+1)];
                prob.blx=[-inf;0;-100*ones(n,1)];
                prob.bux=[inf;0;100*ones(n,1)];
                prob.buc=[inf*ones(counters(T,j),1);hs{1,T}{1,j}];
                prob.blc=[bs{1,T}{1,j}*trial_states{1,T-1}+cs{1,T}{1,j};-inf*ones(counters(T,j),1)];
                prob.a=sparse([subistodcup{1,T}{1,j},subistodcupg{1,T}{1,j}+counters(T,j)],[subjstodcup{1,T}{1,j},subjstodcupg{1,T}{1,j}],[valijstodcup{1,T}{1,j},valijstodcupg{1,T}{1,j}],2*counters(T,j),n+2);
                [~,res]=mosekopt('minimize echo(0)',prob);
                sol=res.sol.itr.xx;
                solsta=strcat('MSK_SOL_STA_', res.sol.itr.solsta);
                if (strcmp(solsta,'MSK_SOL_STA_PRIMAL_INFEASIBLE_CER')==1)
                    disp('Unfeasible primal problem');
                    dual=res.sol.itr.slc(1:counters(T,j));
                    slope=slope+(bs{1,T}{1,j})'*dual;
                    intercept=intercept+sol(1)-dual'*bs{1,T}{1,j}*trial_states{1,T-1};
                    %pause
                elseif (strcmp(solsta,'MSK_SOL_STA_DUAL_INFEASIBLE_CER')==1)
                    disp('Primal infinite optimal value');
                    dual=res.sol.itr.slc(1:counters(T,j));
                    slope=slope+(bs{1,T}{1,j})'*dual;
                    intercept=intercept+sol(1)-dual'*bs{1,T}{1,j}*trial_states{1,T-1};
                    %pause
                else
                    dual=res.sol.itr.slc(1:counters(T,j));
                    slope=slope+(bs{1,T}{1,j})'*dual;
                    intercept=intercept+sol(1)-dual'*bs{1,T}{1,j}*trial_states{1,T-1};
                end
                taux1=taux1+toc;
                xi=xis{1,T}(j,:)';
                ur=Us{1,T}(j);
                psir=Psis{1,T}(j);
                x1=sol(3:n+2);
                x2=trial_states{1,T-1};
                v=x1-x2;
                f1=v'*xi*xi'*v+x1'*xi+1;
                f2=x1'*xi*xi'*x1+x1'*ones(n,1)+ur;
                [vf,indf]=max([f1,f2]);
                g1=4*((x1-ones(n,1))'*(x1-ones(n,1)));
                g2=x1'*xi*xi'*x1+x1'*xi+1;
                [vg,indg]=max([g1,g2]);
                counters(T,j)=counters(T,j)+1;
                if (indf==1)
                    subistodcup{1,T}{1,j}=[subistodcup{1,T}{1,j},counters(T,j)*ones(1,n+1)];
                    subjstodcup{1,T}{1,j}=[subjstodcup{1,T}{j},1,[3:1:n+2]];
                    valijstodcup{1,T}{1,j}=[valijstodcup{1,t}{1,j},1,-2*(x1-x2)'*xi*xi'-xi'];
                    bs{1,T}{1,j}=[bs{1,T}{1,j};2*(x2-x1)'*xi*xi'];
                    cs{1,T}{1,j}=[cs{1,T}{1,j};1-(x1-x2)'*xi*xi'*(x1-x2)];
                else
                    subistodcup{1,T}{1,j}=[subistodcup{1,T}{1,j},counters(T,j)*ones(1,n+1)];
                    subjstodcup{1,T}{1,j}=[subjstodcup{1,T}{1,j},1,[3:1:n+2]];
                    valijstodcup{1,T}{1,j}=[valijstodcup{1,T}{1,j},1,-2*x1'*xi*xi'-ones(1,n)];
                    cs{1,T}{1,j}=[cs{1,T}{1,j};ur-x1'*xi*xi'*x1];
                    bs{1,T}{1,j}=[bs{1,T}{1,j};zeros(1,n)];
                end
                if (indg==1)
                    subistodcupg{1,T}{1,j}=[subistodcupg{1,T}{1,j},counters(t,j)*ones(1,n)];
                    subjstodcupg{1,T}{1,j}=[subjstodcupg{1,T}{1,j},[3:1:n+2]];
                    valijstodcupg{1,T}{1,j}=[valijstodcupg{1,T}{1,j},8*(x1'-ones(1,n))];
                    hs{1,t}{1,j}=[hs{1,T}{1,j};psir+4*(x1-ones(n,1))'*(x1+ones(n,1))];
                else
                    subistodcupg{1,T}{1,j}=[subistodcupg{1,T}{1,j},counters(t,j)*ones(1,n)];
                    subjstodcupg{1,T}{1,j}=[subjstodcupg{1,T}{1,j},[3:1:n+2]];
                    valijstodcupg{1,T}{1,j}=[valijstodcupg{1,T}{1,j},xi'+2*x1'*xi*xi'];
                    hs{1,T}{1,j}=[hs{1,T}{1,j};psir+x1'*xi*xi'*x1-1];
                end
            end
        else
            intercept=0;
            slope=zeros(n,1);
            for j=1:M
                tic
                clear prob;
                prob.c=[1,1,zeros(1,n)];
                prob.blx=[-inf;-10^(10);-100*ones(n,1)];
                prob.bux=[inf;inf;100*ones(n,1)];
                prob.buc=[inf*ones(counters(t,j),1);hs{1,t}{1,j};inf*ones(iter+1,1)];
                prob.blc=[bs{1,t}{1,j}*trial_states{1,t-1}+cs{1,t}{1,j};-inf*ones(counters(t,j),1);thetas{1,t}];
                prob.a=sparse([subia{1,t}+2*counters(t,j),subistodcup{1,t}{1,j},subistodcupg{1,t}{1,j}+counters(t,j)],[subja{1,t},subjstodcup{1,t}{1,j},subjstodcupg{1,t}{1,j}],[valija{1,t},valijstodcup{1,t}{1,j},valijstodcupg{1,t}{1,j}],2*counters(t,j)+iter+1,n+2);
                [~,res]=mosekopt('minimize echo(0)',prob);
                sol=res.sol.itr.xx;
                solsta=strcat('MSK_SOL_STA_', res.sol.itr.solsta);
                if (strcmp(solsta,'MSK_SOL_STA_PRIMAL_INFEASIBLE_CER')==1)
                    disp('Unfeasible primal problem');
                    dual=res.sol.itr.slc(1:counters(t,j));
                    slope=slope+(bs{1,t}{1,j})'*dual;
                    intercept=intercept+sol(1)+sol(2)-dual'*bs{1,t}{1,j}*trial_states{1,t-1};
                    %pause
                elseif (strcmp(solsta,'MSK_SOL_STA_DUAL_INFEASIBLE_CER')==1)
                    disp('Primal infinite optimal value');
                    dual=res.sol.itr.slc(1:counters(t,j));
                    slope=slope+(bs{1,t}{1,j})'*dual;
                    intercept=intercept+sol(1)+sol(2)-dual'*bs{1,t}{1,j}*trial_states{1,t-1};
                    %pause
                else
                    dual=res.sol.itr.slc(1:counters(t,j));
                    slope=slope+(bs{1,t}{1,j})'*dual;
                    intercept=intercept+sol(1)+sol(2)-dual'*bs{1,t}{1,j}*trial_states{1,t-1};
                end
                taux1=taux1+toc;
                xi=xis{1,t}(j,:)';
                ur=Us{1,t}(j);
                psir=Psis{1,t}(j);
                x1=sol(3:n+2);
                x2=trial_states{1,t-1};
                v=x1-x2;
                f1=v'*xi*xi'*v+x1'*xi+1;
                f2=x1'*xi*xi'*x1+x1'*ones(n,1)+ur;
                [vf,indf]=max([f1,f2]);
                g1=4*((x1-ones(n,1))'*(x1-ones(n,1)));
                g2=x1'*xi*xi'*x1+x1'*xi+1;
                [vg,indg]=max([g1,g2]);
                counters(t,j)=counters(t,j)+1;
                if (indf==1)
                    subistodcup{1,t}{1,j}=[subistodcup{1,t}{1,j},counters(t,j)*ones(1,n+1)];
                    subjstodcup{1,t}{1,j}=[subjstodcup{1,t}{1,j},1,[3:1:n+2]];
                    valijstodcup{1,t}{1,j}=[valijstodcup{1,t}{1,j},1,-2*(x1-x2)'*xi*xi'-xi'];
                    bs{1,t}{1,j}=[bs{1,t}{1,j};2*(x2-x1)'*xi*xi'];
                    cs{1,t}{1,j}=[cs{1,t}{1,j};1-(x1-x2)'*xi*xi'*(x1-x2)];
                else
                    subistodcup{1,t}{1,j}=[subistodcup{1,t}{1,j},counters(t,j)*ones(1,n+1)];
                    subjstodcup{1,t}{1,j}=[subjstodcup{1,t}{1,j},1,[3:1:n+2]];
                    valijstodcup{1,t}{1,j}=[valijstodcup{1,t}{1,j},1,-2*x1'*xi*xi'-ones(1,n)];
                    cs{1,t}{1,j}=[cs{1,t}{1,j};ur-x1'*xi*xi'*x1];
                    bs{1,t}{1,j}=[bs{1,t}{1,j};zeros(1,n)];
                end
                if (indg==1)
                    subistodcupg{1,t}{1,j}=[subistodcupg{1,t}{1,j},counters(t,j)*ones(1,n)];
                    subjstodcupg{1,t}{1,j}=[subjstodcupg{1,t}{1,j},[3:1:n+2]];
                    valijstodcupg{1,t}{1,j}=[valijstodcupg{1,t}{1,j},8*(x1'-ones(1,n))];
                    hs{1,t}{1,j}=[hs{1,t}{1,j};psir+4*(x1-ones(n,1))'*(x1+ones(n,1))];
                else
                    subistodcupg{1,t}{1,j}=[subistodcupg{1,t}{1,j},counters(t,j)*ones(1,n)];
                    subjstodcupg{1,t}{1,j}=[subjstodcupg{1,t}{1,j},[3:1:n+2]];
                    valijstodcupg{1,t}{1,j}=[valijstodcupg{1,t}{1,j},xi'+2*x1'*xi*xi'];
                    hs{1,t}{1,j}=[hs{1,t}{1,j};psir+x1'*xi*xi'*x1-1];
                end
            end
        end
        slope=slope/M;
        intercept=intercept/M;
        subia{1,t-1}=[subia{1,t-1},(iter+1)*ones(1,n+1)];
        subja{1,t-1}=[subja{1,t-1},[2:1:n+2]];
        valija{1,t-1}=[valija{1,t-1},1,-slope'];
        thetas{1,t-1}=[thetas{1,t-1};intercept];
    end
    else
        for t=T:-1:2
            intercept=0;
            slope=zeros(n,1);
            for j=1:M
                tic
                xi=xis{1,t}(j,:)';
                ur=Us{1,t}(j);
                psir=Psis{1,t}(j);
                clear prob;
                prob.qcsubk=[];
                prob.qcsubi=[];
                prob.qcsubj=[];
                prob.qcval=[];
                subi=[];
                subj=[];
                valij=[];
                prob.c=[1,1,zeros(1,2*n+1)];
                if (t==T)
                    prob.blx=[-inf;0;-100*ones(n,1);-inf*ones(n+1,1)];
                    prob.bux=[inf;0;100*ones(n,1);inf*ones(n+1,1)];
                else
                    prob.blx=[-inf;-10^(10);-100*ones(n,1);-inf*ones(n+1,1)];
                    prob.bux=[inf;inf;100*ones(n,1);inf*ones(n+1,1)];
                end
                if (t==T)
                    prob.buc=[-1;-ur;-4*n;-1;psir;trial_states{1,t-1}];
                    prob.blc=[-inf;-inf;-inf;-inf;-inf;trial_states{1,t-1}];
                else
                    prob.buc=[-1;-ur;-4*n;-1;psir;inf*ones(iter+1,1);trial_states{1,t-1}];
                    prob.blc=[-inf;-inf;-inf;-inf;-inf;thetas{1,t};trial_states{1,t-1}];
                end

                %Constraint 1
                subi=[subi,ones(1,n+1)];
                subj=[subj,1,[3:1:n+2]];
                valij=[valij,-1,xi'];
                for i=1:n
                    for k=1:i
                        prob.qcsubk=[prob.qcsubk,1];
                        prob.qcsubi=[prob.qcsubi,2+i];
                        prob.qcsubj=[prob.qcsubj,2+k];
                        prob.qcval=[prob.qcval,2*xi(i)*xi(k)];
                    end
                end
                for i=1:n
                    for k=1:i
                        prob.qcsubk=[prob.qcsubk,1];
                        prob.qcsubi=[prob.qcsubi,3+n+i];
                        prob.qcsubj=[prob.qcsubj,3+n+k];
                        prob.qcval=[prob.qcval,2*xi(i)*xi(k)];
                    end
                end
                for i=1:n
                    for k=1:n
                        prob.qcsubk=[prob.qcsubk,1];
                        prob.qcsubi=[prob.qcsubi,3+n+i];
                        prob.qcsubj=[prob.qcsubj,3+n+k];
                        prob.qcval=[prob.qcval,-2*xi(i)*xi(k)];
                    end
                end
               
                %Constraint 2
                subi=[subi,2*ones(1,n+1)];
                subj=[subj,1,[3:1:n+2]];
                valij=[valij,-1,ones(1,n)];
                for i=1:n
                    for k=1:i
                        prob.qcsubk=[prob.qcsubk,2];
                        prob.qcsubi=[prob.qcsubi,2+i];
                        prob.qcsubj=[prob.qcsubj,2+k];
                        prob.qcval=[prob.qcval,2*xi(i)*xi(k)];
                    end
                end
                
                %Constraint 3
                subi=[subi,3*ones(1,n+1)];
                subj=[subj,n+3,[3:1:n+2]];
                valij=[valij,-1,-8*ones(1,n)];
                for i=1:n
                    prob.qcsubk=[prob.qcsubk,3];
                    prob.qcsubi=[prob.qcsubi,2+i];
                    prob.qcsubj=[prob.qcsubj,2+i];
                    prob.qcval=[prob.qcval,8];
                end
                
                %Constraint 4
                subi=[subi,4*ones(1,n+1)];
                subj=[subj,n+3,[3:1:n+2]];
                valij=[valij,-1,xi'];
                for i=1:n
                    for k=1:i
                        prob.qcsubk=[prob.qcsubk,4];
                        prob.qcsubi=[prob.qcsubi,2+i];
                        prob.qcsubj=[prob.qcsubj,2+k];
                        prob.qcval=[prob.qcval,2*xi(i)*xi(k)];
                    end
                end
            
                %Constraint 5
                subi=[subi,5];
                subj=[subj,n+3];
                valij=[valij,1];
            
                if (t<T)
                   subi=[subi,5+subia{1,t}];
                   subj=[subj,subja{1,t}];
                   valij=[valij,valija{1,t}];
                
                   for i=1:n
                       subi=[subi,5+iter+1+i];
                       subj=[subj,n+3+i];
                       valij=[valij,1];
                   end
                else
                   for i=1:n
                       subi=[subi,5+i];
                       subj=[subj,n+3+i];
                       valij=[valij,1];
                   end
                end
            
               if (t==T)
                   prob.a=sparse(subi,subj,valij,5+n,2*n+3);
               else
                   prob.a=sparse(subi,subj,valij,5+iter+1+n,2*n+3);
               end
               [~,res]=mosekopt('minimize',prob);
               sol=res.sol.itr.xx;
               solsta=strcat('MSK_SOL_STA_', res.sol.itr.solsta);
               if (strcmp(solsta,'MSK_SOL_STA_PRIMAL_INFEASIBLE_CER')==1)
                  disp('Unfeasible primal problem');
                  if (t==T)
                      dual=res.sol.itr.slc(5+iter+1:5+iter+n)-res.sol.itr.suc(5+iter+1:5+iter+n);
                  else
                      dual=res.sol.itr.slc(5+iter+2:5+iter+1+n)-res.sol.itr.suc(5+iter+2:5+iter+1+n);
                  end
                  slope=slope+dual;
                  intercept=intercept+sol(1)+sol(2)-dual'*trial_states{1,t-1};
                  %pause
               elseif (strcmp(solsta,'MSK_SOL_STA_DUAL_INFEASIBLE_CER')==1)
                  disp('Primal infinite optimal value');
                  if (t==T)
                      dual=res.sol.itr.slc(5+iter+1:5+iter+n)-res.sol.itr.suc(5+iter+1:5+iter+n);
                  else
                      dual=res.sol.itr.slc(5+iter+2:5+iter+1+n)-res.sol.itr.suc(5+iter+2:5+iter+1+n);
                  end
                  slope=slope+dual;
                  intercept=intercept+sol(1)+sol(2)-dual'*trial_states{1,t-1};
                  %pause
               else
                  if (t==T)
                      dual=res.sol.itr.slc(5+iter+1:5+iter+n)-res.sol.itr.suc(5+iter+1:5+iter+n);
                  else
                      dual=res.sol.itr.slc(5+iter+2:5+iter+1+n)-res.sol.itr.suc(5+iter+2:5+iter+1+n);
                  end
                  slope=slope+dual;
                  intercept=intercept+sol(1)+sol(2)-dual'*trial_states{1,t-1};
               end
               taux1=taux1+toc;
        end
        slope=slope/M;
        intercept=intercept/M;
        subia{1,t-1}=[subia{1,t-1},(iter+1)*ones(1,n+1)];
        subja{1,t-1}=[subja{1,t-1},[2:1:n+2]];
        valija{1,t-1}=[valija{1,t-1},1,-slope'];
        thetas{1,t-1}=[thetas{1,t-1};intercept];        
        end
    end
    time_sddp=[time_sddp;taux1];
    End_Algo=abs((zsup-zinf)/zsup)>tol;
    if (iter>=nb_iter_max)
        End_Algo=0;
    end
    iter=iter+1;
end



