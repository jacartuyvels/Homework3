%% Homework 3 Optimization run.m 
%% Exercise 1

data = load('Homework-3-data.mat');

training = data.training;
test = data.test;


label = training.labels;
image = training.images;
shape = size(image);

image = reshape(image,[shape(1)*shape(2), shape(3)]);
[data0 data1] = partition(image,label,0);

nA = size(data0,2);
nB = size(data1,2);
data0 = data0(:,1:200);
data1 = data1(:,1:1200);


nIter = 50;
L = 10;
lambda = 10^(-6);
lambda2 = 0.01
R = 8

x = zeros(size(data0,1)+1,1);

%[xOut1 gradOut1] = gradient1(x,nIter,R,L,data0,data1);
%[xOut2 gradOut2] = gradient2(x,nIter,lambda,L,data0,data1);
%[xOut3 gradOut3] = accGradient(x,nIter,L,lambda,data0,data1);
[xOut4 gradOut4] = subgradient(x,nIter,L,lambda2,data0,data1);

%%
nA = size(data0,2);
nB = size(data1,2);
objective = zeros(nIter,4);

for j = 1:nIter    
    for i = 1:nA
        y1 = xOut1(1:end-1,j)'*data0(:,i) +xOut1(end,j);
        objective(j,1) = objective(j,1) + log(1+exp(-y1))/nA;
    end

    
    for i = 1:nB
        y2 = xOut1(1:end-1,j)'*data1(:,i) + xOut1(end,j);
        objective(j,1) = objective(j,1) + log(1+exp(y2))/nB;
    end    
    
end
m = 1:nIter;
worst1 = L./(2*m)*norm(xOut1(:,1)-xOut1(:,end))^2;

for j = 1:nIter    
    for i = 1:nA
        y1 = xOut2(1:end-1,j)'*data0(:,i) +xOut2(end,j);
        objective(j,2) = objective(j,2) + log(1+exp(-y1))/nA;
    end

    
    for i = 1:nB
        y2 = xOut2(1:end-1,j)'*data1(:,i) + xOut2(end,j);
        objective(j,2) = objective(j,2) + log(1+exp(y2))/nB;
    end
    objective(j,2) = objective(j,2) +   lambda/2 *norm(xOut2(1:end-1,j));
end

worst2 = L./(2*m)*norm(xOut2(:,1)-xOut2(:,end))^2;



for j = 1:nIter    
    for i = 1:nA
        y1 = xOut3(1:end-1,j)'*data0(:,i) +xOut3(end,j);
        objective(j,3) = objective(j,3) + log(1+exp(-y1))/nA;
    end

    
    for i = 1:nB
        y2 = xOut3(1:end-1,j)'*data1(:,i) + xOut3(end,j);
        objective(j,3) = objective(j,3) + log(1+exp(y2))/nB;
    end
    objective(j,3) = objective(j,3) +   lambda/2 *norm(xOut3(1:end-1,j));
end

worst3 = 2*L./(m).^2*norm(xOut2(:,1)-xOut2(:,end))^2;

%%


subplot(2,2,2)
semilogy(m,objective(:,1),m,worst1)
title('Gradient with R-bounded constraint')
legend('Our method','Worst case')
subplot(2,2,3)
semilogy(m,objective(:,2),m,worst2)
title('Gradient with lambda-regularized variable')
legend('Our method','Worst case')
subplot(2,2,4)
semilogy(m,objective(:,3),m,worst3)
title('Accelerated gradient with lambda-regularized variable')
legend('Our method','Worst case')


