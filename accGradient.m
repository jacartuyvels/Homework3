%% Compute the accelerated gradient méthod on logistic loss function and lambda regularized variable
function [xOut gradOut] = accGradient(x,nIter,L,lambda,data0,data1)

gradOut = [];
xOut= [x];
sizeH = size(data0,1);
nA = size(data0,2);
nB = size(data1,2);

n = 0;

xnew = x;

while n < nIter 
    
    gradH = zeros(sizeH,1);
    gradC = 0;
    if n==0
        normH = 1;
        y0 = xnew;
    else 
        beta = (n-2)/(n+1);
        y0 = xOut(:,end) + beta*(xOut(:,end)-xOut(:,end-1));
        normH = norm(y0(1:end-1))
    end
        
    for j = 1:nA
        expA = exp(data0(:,j)'*y0(1:sizeH)+y0(end));
        gradH = gradH + (-data0(:,j)/(1+expA))/nA + lambda*y0(1:end-1)/normH;
        gradC = gradC + (1/(1+expA))/nA ;
    end
    
    for j = 1:nB
        expB = exp(-data1(:,j)'*y0(1:sizeH)-y0(end));
        gradH = gradH + (data1(:,j)/(1+expB))/nB + lambda*y0(1:end-1)/normH;
        gradC = gradC + (1/(1+expB))/nB;
    end
    
           
    

    xnew(1:sizeH) = y0(1:sizeH)-(1/L)*(gradH);
    xnew(end) = y0(end)-(1/L)*(gradC);

    gradOut = [gradOut norm([gradH;gradC])];
    xOut = [xOut xnew];
 
    n = n+1
end

x= xnew;

end



