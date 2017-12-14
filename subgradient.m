function xOut = subgradient(x,nIter,L,lamda,data0,data1)
%initialisation de xOut
xOut= [];
%taille des données
shape1 = size(data0);
shape2 = size(data1);
sizeH = shape1(1);
nA = shape1(2);
nB = shape2(2);
%nombre d'iteration
n = 0;
xnew = x; %xnew = valeur actuel de x
while n < nIter
    %initialisation des gradients
    gradH = zeros(sizeH,1);
    gradC = 0;
    
    %initialisation de Xk+1
    x0 = xnew; % on va trouver un nouveau x0, on l'initilise a xnew
    
    if n==0
        normH = 1;
    else 
        normH = norm(x0(1:end-1));
    end
    
    for j = 1:nA
        zA = data0(:,j)'*x0(1:sizeH)+x0(end);
        if 1-zA < 0
            gradH = gradH + lamda*x0(1:end-1)/normH;
            gradC = gradC;
        elseif 1-zA > 0
            gradH = gradH - data0(:,j) + lamda*x0(1:end-1)/normH;
            gradC = gradC - 1;
        else
            gradH = gradH - 1/2*data0(:,j) + lamda*x0(1:end-1)/normH;
            gradC = gradC - 1/2;
        end
    end
    
    for j = 1:nB
        zB = -data1(:,j)'*x0(1:sizeH)-x0(end);
        if 1-zB < 0
            gradH = gradH + lamda*x0(1:end-1)/normH;
            gradC = gradC;
        elseif 1-zB > 0
            gradH = gradH + data1(:,j) + lamda*x0(1:end-1)/normH;
            gradC = gradC + 1;
        else
            gradH = gradH + 1/2 * data1(:,j) + lamda*x0(1:end-1)/normH;
            gradC = gradC + 1/2;
        end
    end
    xnew(1:sizeH) = x0(1:sizeH)-(1/L)*(gradH);
    xnew(end) = x0(end)-(1/L)*(gradC);
    
    xOut = [xOut xnew];
    %%%%%
    n = n +1
end
end