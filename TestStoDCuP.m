addpath 'C:\Program Files\Mosek\9.0'
addpath 'C:\Program Files\Mosek\9.0\toolbox\r2015a'
addpath 'C:\Users\vince\Dropbox\ISDDP_NLP';
T=5;
M=20;
Ms=[1;M*ones(T-1,1)];
n=10;
[xis,Us,Psis]=generate_scenarios(Ms,T,n);
nb_iter_max=10;
tol=0.1;
x0=ones(n,1);
Minit=20;
probabilities=cell(1,T-1);
alpha=0.2;
for t=1:T-1
    probabilities{1,t}=(1/M)*ones(1,M);
end
accuracies=[10*ones(10,1);5*ones(10,1);3*ones(20,1);ones(100,1);0.5*ones(100,1);0.1*ones(110,1);(10^(-6))*ones(2660,1)];

%SDDP-ND
accuracies=[(10^(-10))*ones(1,3000)];
iter_stodcup=0;
[lower_bounds_sddp_nd,upper_bounds_sddp_nd,time_sddp_nd,nb_f_unsolved,nb_b_unsolved]=inexact_stodcup_quadratic(T,n,M,iter_stodcup,nb_iter_max,xis,Us,Psis,tol,x0,Minit,probabilities,accuracies,alpha);



%ISDDP-D
iter_inexact=0;
alpha=0;
[lower_bounds_isddp_d,upper_bounds_isddp_d,time_isddp_d,nb_f_unsolved,nb_b_unsolved]=inexact_sddp_quadratic(T,n,M,iter_inexact,nb_iter_max,xis,Us,Psis,tol,x0,probabilities,accuracies,alpha);

%SDDP Diff
accuracies=[10^(-12)*ones(1,3000)];
iter_inexact=0;
[lower_bounds_sddp_d,upper_bounds_sddp_d,time_sddp_d,nb_f_unsolved,nb_b_unsolved]=inexact_sddp_quadratic(T,n,M,iter_inexact,nb_iter_max,xis,Us,Psis,tol,x0,probabilities,accuracies,alpha);

%ISDDP-ND
accuracies=[10*ones(10,1);5*ones(10,1);3*ones(20,1);ones(100,1);0.5*ones(100,1);0.1*ones(110,1);(10^(-6))*ones(2660,1)];
iter_stodcup=0;
[lower_bounds_isddp_nd,upper_bounds_isddp_nd,time_isddp_nd,nb_f_unsolved,nb_b_unsolved]=inexact_stodcup_quadratic(T,n,M,iter_stodcup,nb_iter_max,xis,Us,Psis,tol,x0,Minit,probabilities,accuracies,alpha);


%StoDCuP
accuracies=[(10^(-6))*ones(1,3000)];
iter_stodcup=2000;
[lower_bounds_scup,upper_bounds_scup,time_scup,nb_f_unsolved,nb_b_unsolved]=inexact_stodcup_quadratic(T,n,M,iter_stodcup,nb_iter_max,xis,Us,Psis,tol,x0,Minit,probabilities,accuracies,alpha);

%IStoDCuP
accuracies=[10*ones(10,1);5*ones(10,1);3*ones(20,1);ones(100,1);0.5*ones(100,1);0.1*ones(110,1);(10^(-6))*ones(2660,1)];
iter_stodcup=1000;
[lower_bounds_iscup,upper_bounds_iscup,time_iscup,nb_f_unsolved,nb_b_unsolved]=inexact_stodcup_quadratic(T,n,M,iter_stodcup,nb_iter_max,xis,Us,Psis,tol,x0,Minit,probabilities,accuracies,alpha);

%Mixed IStoDCuP-SDDP
accuracies=[10*ones(10,1);5*ones(10,1);3*ones(20,1);ones(100,1);0.5*ones(100,1);0.1*ones(110,1);(10^(-6))*ones(2660,1)];
iter_stodcup=5;
[lower_bounds_miscup,upper_bounds_miscup,time_miscup,nb_f_unsolved,nb_b_unsolved]=inexact_stodcup_quadratic(T,n,M,iter_stodcup,nb_iter_max,xis,Us,Psis,tol,x0,Minit,probabilities,accuracies,alpha);



%Test of consistency of the choice of objective and constraint functions

for t=1:T
    t
    for j=1:Ms(t)
        j
        xi=xis{1,t}(j,:)';
        uf=Us{1,t}(j);
        psc=Psis{1,t}(j);
        fo=[];
        fc=[];
        for i=1:10000
                x1=-100*ones(n,1)+200*rand(n,1);
                x2=-100*ones(n,1)+200*rand(n,1);
                v=x1-x2;
                f1=v'*xi*xi'*v+x1'*xi+1;
                f2=x1'*xi*xi'*x1+x1'*ones(n,1)+uf;
                fo=[fo;f1-f2];
                f1c=4*((x1-ones(n,1))'*(x1-ones(n,1)));
                f2c=x1'*xi*xi'*x1+x1'*xi+1;
                fc=[fc;f1c-f2c];
        end
        subplot(1,2,1);
        plot(fo);
        subplot(1,2,2);
        plot(fc);
        hold on
        subplot(1,2,2);
        plot(psc*ones(1,10000),'r');
        pause
    end
end

for t=1:T
    t
    for j=1:Ms(t)
        j
        xi=xis{1,t}(j,:)';
        uf=Us{1,t}(j);
        psc=Psis{1,t}(j);
        fc=[];
        for i=1:10000
                x1=-100*ones(n,1)+200*rand(n,1);
                x2=-100*ones(n,1)+200*rand(n,1);
                v=x1-x2;
                f1c=4*((x1-ones(n,1))'*(x1-ones(n,1)));
                f2c=x1'*xi*xi'*x1+x1'*xi+1;
                fc=[fc;f1c-f2c];
        end
        plot(fc);
        hold on
        plot(psc*ones(1,10000),'r');
        pause
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

%Graph of f.

T=3;
Ms=[1;20*ones(T-1,1)];
n=10;
addpath 'C:\Users\vince\Dropbox\Softwares\StoDCuP';
[xis,Us,Psis]=generate_scenarios(Ms,T,n);

for t=1:T
    t
    for j=1:Ms(t)
        j
        xi=xis{1,t}(j,:)';
        uf=Us{1,t}(j);
        psc=Psis{1,t}(j);
        minf=inf;
        for i=1:10000
                x1=-100*ones(n,1)+200*rand(n,1);
                x2=-100*ones(n,1)+200*rand(n,1);
                v=x1-x2;
                f1=v'*xi*xi'*v+x1'*xi+1;
                f2=x1'*xi*xi'*x1+x1'*ones(n,1)+uf;
                if (max(f1,f2)<minf)
                    minf=max(f1,f2);
                    argminf=[x1;x2];
                end
        end
        minf
        argminf
        pause
    end
end



