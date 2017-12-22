%% Compute the gradient m�thod with logistic loss and lambda regularized variable
function [xOut gradOut] = gradient2(x,nIter,lambda,L,data0,data1)



xOut= [];
gradOut = [];
sizeH = size(data0,1);
nA = size(data0,2);
nB = size(data1,2);
% L = 0;
% 
% for i = 1:nA
%     h = data0(:,i);
%     L = L + 1/4*norm([h; 1]*[h' 1],inf)/nA;
% end
% 
% for i = 1:nB
%     h = data1(:,i);
%     L = L + 1/4*norm([h ;1]*[h' 1],inf)/nB;
% end

n = 0;

xnew = x;

while n < nIter 
    x0 = xnew;
    gradH = zeros(sizeH,1);
    gradC = 0;
    if n==0
        normH = 1;
    else 
        normH = norm(x0(1:end-1));
    end
        
    for j = 1:nA
        expA = exp(data0(:,j)'*x0(1:sizeH)+x0(end));
        gradH = gradH + (-data0(:,j)/(1+expA))/nA + lambda*x0(1:end-1)/normH;
        gradC = gradC + (1/(1+expA))/nA ;
    end
    
    for j = 1:nB
        expB = exp(-data1(:,j)'*x0(1:sizeH)-x0(end));
        gradH = gradH + (data1(:,j)/(1+expB))/nB + lambda*x0(1:end-1)/normH;
        gradC = gradC + (1/(1+expB))/nB;
    end
    
           
    

    xnew(1:sizeH) = x0(1:sizeH)-(1/L)*(gradH);

    xnew(end) = x0(end)-(1/L)*(gradC);
    
    
    gradOut = [gradOut norm([gradH;gradC])];
    xOut = [xOut xnew];

    n = n+1
end
x= xnew;

end



