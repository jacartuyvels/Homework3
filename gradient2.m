%% Compute the gradient méthod with logistic loss and lambda regularized variable
function xOut = gradient1(x,nIter,L,lambda,data0,data1)




shape1 = size(data0);
shape2 = size(data1);

xOut= [];
sizeH = shape1(1);
nA = shape1(2);
nB = shape2(2);


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
    
    
    
    xOut = [xOut xnew];

    n = n+1
end

x= xnew;

end



