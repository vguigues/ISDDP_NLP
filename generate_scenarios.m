
%Scenarios:
%xis{1,t}(k,j) scenario k component j for stage t.
%Us{1,t}(k) 
%Psis{1,t}(k)

function [xis,Us,Psis]=generate_scenarios(Ms,T,n)

xis=cell(1,T);
Us=cell(1,T);
Psis=cell(1,T);

for t=1:T
    xis{1,t}=zeros(Ms(t),n);
    Us{1,t}=zeros(1,Ms(t));
    Psis{1,t}=zeros(1,Ms(t));
    A=zeros(n,n);
    for k=1:Ms(t)
        for i=1:n
            for j=1:n
                A(i,j)=-0.5+rand;
            end
        end
        Sigma=A*A'+0.5*eye(n);
        moy=zeros(1,n);
        for i=1:n
            u=rand;
            if (u<=0.5)
                moy(i)=-1;
            else
                moy(i)=1;
            end
        end
        xis{1,t}(k,:)=moy+((Sigma^(0.5))*randn(n,1))';
        u=rand;
        if (u<=0.5)
            Us{1,t}(k)=-10;
        else
            Us{1,t}(k)=10;
        end
        Psis{1,t}(k)=10000+(100000-10000)*rand;
    end
end


