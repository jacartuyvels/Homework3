%% Compute the gradient méthod with logistic loss and R bounded variable
function xOut = gradient1(x,nIter,L,R,data0,data1)




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
    
    xOut = [xOut xnew];

    n = n+1
end

x= xnew;

end



