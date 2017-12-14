function [h, c, tau] = marginLin(data0, data1)
% Find a maximum margin separating hyperplane betweens sets of points pa et pb
% (data points are arranged as columns in matrices pa and pb)
% Model: min t s.t. h^T ai + c <= -1, h^t bi + c >= 1 and norm(h) <= t
% (alternative model: max t s.t. h^T ai + c <= -t, h^t bi + c >= t and norm(h) <= 1)


nA = size(data0,2);
nB = size(data1,2);
n = size(data0,1);

A = [-data0' -ones(nA,1) ones(nA,1);data1' ones(nB,2); zeros(1,n+2); eye(n) zeros(n,2)]';

c = [zeros(nA+nB,1); 1; zeros(n,1)];

b = [zeros(n+1,1); 1];

K.l = nA+nB;
K.q = n+1;

[x,y,info]=sedumi(A, b, c, K);
h = y(1:n)
c = y(n+1)
tau = y(n+2)


if info.dinf ==1
    disp('There exists no separating hyperplane');
    h = []; c = [];
else
    disp(['Separating hyperplane: margin=' num2str(tau,3) ' ; h=' mat2str(h,3) ' ; c=' num2str(c,3)]);
end