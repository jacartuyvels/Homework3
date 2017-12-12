%% Compute the gradient méthod with logistic loss and R bounded variable
function [x,tot] = gradient1(x,tol,nIter,R,im,label)

L = 0.001;
shape = size(im);
tot = zeros(nIter+1,1);
sizeH = shape(1);
delta = 10;
n = 0;



xnew = x;
while n < nIter && tol < abs(delta)
    x0 = xnew;
    gradA = zeros(sizeH,1);
    gradB = zeros(sizeH,1);
    gradCA = 0;
    gradCB = 0;
    gradLambda = 0
    nA = 0;
    nB = 0;
    yA = 0;
    yB = 0;
    normH = norm(x(1:sizeH))
    for i =1:shape(2)
        if label(i) == 1
            nA = nA + 1;
            expA = exp(im(:,i)'*x0(1:sizeH)+x0(end));
            yA = log(1+exp(-im(:,1)'*x0(1:sizeH)-x0(end)));
            gradLambda = gradLambda + x(sizeH+2)*sum(x(1:sizeH))/(2*normH);
            gradA = gradA + -im(:,i)/(1+expA);
            gradCA = gradCA + 1/(1+expA) ;
       else
            nB = nB + 1;
            expB = exp(-im(:,i)'*x0(1:sizeH)-x0(end));
            gradB = gradB + im(:,i)/(1+expB);
            yB = log(1+exp(x0(1:sizeH)'*im(:,i)+x0(end)));
            gradLambda = gradLambda + x(sizeH+2)*sum(x(1:sizeH))/(2*normH);
            gradCB = gradCB + 1/(1+expB);
        end
    end

    gradH = gradA/nA + gradB/nB;
    gradC = gradCA/nA + gradCB/nB;
    tot(n+1,1) = yA/nA + yB/nB;

    xnew(1:sizeH) = x0(1:sizeH)-(1/L)*(gradH);
    
    if norm(xnew(1:sizeH))> R
        xnew(1:sizeH) = xnew(1:sizeH)/norm(xnew(1:sizeH))*R;
    end


    xnew(end) = x0(end)-(1/L)*(gradC);
    %delta = norm(x0-xnew)
    n = n+1
    delta = 1;
end

% Computation of the final y
for i =1:shape(2)
    if label(i) == 1
        nA = nA + 1;
        yA = log(1+exp(-x0(1:sizeH)'*im(:,i)-x0(end)));
    else
        nB = nB + 1;
        yB = log(1+exp(x0(1:sizeH)'*im(:,i)+x0(end)));
    end
end
tot(nIter+1) = yA/nA + yB/nB;
x= xnew;

end



