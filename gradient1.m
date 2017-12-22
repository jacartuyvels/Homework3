%% Compute the gradient méthod with logistic loss and R bounded variable
function [xOut gradOut] = gradient1(x,nIter,R,L,data0,data1)


xOut= [];
sizeH = size(data0,1);
nA = size(data0,2);
nB = size(data1,2);
gradOut = [];

% for i = 1:nA
%     h = data0(:,i);
%     L = L + 1/4*norm([h; 1]*[h' 1],2)/nA;
% end
% 
% for i = 1:nB
%     h = data1(:,i);
%     L = L + 1/4*norm([h ;1]*[h' 1],2)/nB;
% end


n = 0;

xnew = x;

while n < nIter 
    x0 = xnew;
    gradH = zeros(sizeH,1);
    gradC = 0;

    for j = 1:nA
        expA = exp(data0(:,j)'*x0(1:sizeH)+x0(end));
        gradH = gradH + (-data0(:,j)/(1+expA))/nA;
        gradC = gradC + (1/(1+expA))/nA ;
    end
    
    for j = 1:nB
        expB = exp(-data1(:,j)'*x0(1:sizeH)-x0(end));
        gradH = gradH + (data1(:,j)/(1+expB))/nB;
        gradC = gradC + (1/(1+expB))/nB;
    end
    
           
    

    xnew(1:sizeH) = x0(1:sizeH)-(1/L)*(gradH);

    xnew(end) = x0(end)-(1/L)*(gradC);
    
    if norm(xnew)> R
        xnew = xnew/norm(xnew)*R;
    end
    newgrad = norm([gradH;gradC]);
    gradOut = [gradOut newgrad];
    xOut = [xOut xnew];

    n = n+1
end

x= xnew;
L
end



