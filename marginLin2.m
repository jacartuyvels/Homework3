function [h, c] = marginLin2(data0, data1,lambda)


nA = size(data0,2);
nB = size(data1,2);
n = size(data0,1);

%Dual
A1 = [data0' ones(nA,1) eye(nA) zeros(nA,nB) zeros(nA,1)];
A2 = [-data1' -ones(nB,1) zeros(nB,nA) eye(nB) zeros(nB,1)];
A3 = [zeros(nA,n+1) eye(nA) zeros(nA,nB+1)];
A4 = [zeros(nB,n+nA+1) eye(nB) zeros(nB,1)];
A5 = [zeros(1,n+1+nA+nB) 1; zeros(1,n+2+nA+nB); eye(n) zeros(n,2+nA+nB)];
A = -[A1;A2;A3;A4;A5]';
c = [-ones(nA+nB,1); zeros(nA+nB,1); 0; 1 ;zeros(n,1)];
b = -[zeros(n+1,1); ones(nA+nB,1); lambda/2];

K.l = 2*(nA+nB);
K.r = n+2;

[x,y,info]=sedumi(A, b, c, K);

h = y(1:n);
c = y(n+1);


%Primal
% A1 = [ones(nA,1) -eye(nA) zeros(nA,nB) eye(nA) zeros(nA,nB) zeros(nA,2) data0'];
% A2 = [-ones(nB,1) zeros(nB,nA) -eye(nB) zeros(nB,nA) eye(nB) zeros(nB,2) data1'];
% A3 = [zeros(1,2+2*nA+2*nB) 1 zeros(1,n)];
% A = [A1;A2;A3];
% 
% b =  [ones(nA+nB+1,1)];
% 
% c = [zeros(1+nA+nB,1); ones(nA+nB+1,1); zeros(n+1,1)];
% 
% K.f = 1;
% K.l =2*nA+2*nB;
% K.r = 2+n;
% 
% [x,y,info]=sedumi(A, b, c, K);
% 
% h = x(end-n+1:end);
% c = x(1);

if info.dinf ==1
    disp('There exists no separating hyperplane');
    h = []; c = [];
else
    disp(['Separating hyperplane: h=' mat2str(h,3) ' ; c=' num2str(c,3)]);
end