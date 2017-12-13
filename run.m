%% Homework 3 Optimization run.m 
%% Exercise 1

data = load('Homework-3-data.mat');

training = data.training;
test = data.test;



label = training.labels;
image = training.images;
shape = size(image);

image = reshape(image(:,:,:),[shape(1)*shape(2), shape(3)]);
[data0 data1] = partition(image,label,0);

data0 = data0(:,1:2000);
data1 = data1(:,1:12000);


%% 1.2 Gradient descent on logistic loss function with R-bounded variable 
 
% variable [h_i c]' for i=1, ... , 28^2



tol = 0;
nIter = 200;
R = 8;
x= zeros(shape(1)*shape(2)+1,1);
xOut = zeros(shape(1)*shape(2)+1,nIter);
L = 10;

xOut = gradient1(x,nIter,L,R,data0,data1);



nA = size(data0);
nB = size(data1);

y = zeros(nA(2),nIter);
y2 = zeros(nB(2),nIter);
accuracy1 = zeros(nIter,1);
accuracy2 = zeros(nIter,2);

for j = 1:nIter
    
    for i = 1:nA(2)
        y(i,j) = xOut(1:end-1,j)'*data0(:,i) + xOut(end,j);
        
        accuracy1(j) = accuracy1(j) + max(0,y(i,j))/y(i,j);
        
    end
    
    accuracy1(j) = accuracy1(j)/nA(2) ;
    
    for i = 1:nB(2)
        y2(i,j) = xOut(1:end-1,j)'*data1(:,i) + xOut(end,j) ; 
        
        accuracy2(j) = accuracy2(j) + min(0,y2(i,j))/y2(i,j);
        
    end
    accuracy2(j) = accuracy2(j)/nB(2);
end
 
hold off
plot(accuracy1)
hold on 
plot(accuracy2)
title('Accuracy of gradient descent on logistic loss function with R-bounded variable')
legend('Accuracy of 0','Accuracy if not 0')
xlabel('Number of iteration')
ylabel('Accuracy')
    
    
%% 1.3 Gradient descent on logistic loss function with Lambda regularized objective

% variable [h_i c]' for i=1, ... , 28^2

nIter = 20;
lambda = 0.0000001;
x= zeros(shape(1)*shape(2)+1,1);
xOut = zeros(shape(1)*shape(2)+1,nIter);
L = 10

xOut = gradient2(x,nIter,L,lambda,data0,data1);



nA = size(data0);
nB = size(data1);

y = zeros(nA(2),nIter);
y2 = zeros(nB(2),nIter);
accuracy1 = zeros(nIter,1);
accuracy2 = zeros(nIter,1);

for j = 1:nIter
    
    for i = 1:nA(2)
        y(i,j) = xOut(1:end-1,j)'*data0(:,i) + xOut(end,j);
        
        accuracy1(j) = accuracy1(j) + max(0,y(i,j))/y(i,j);
        
    end
    
    accuracy1(j) = accuracy1(j)/nA(2) ;
    
    for i = 1:nB(2)
        y2(i,j) = xOut(1:end-1,j)'*data1(:,i) + xOut(end,j) ; 
        
        accuracy2(j) = accuracy2(j) + min(0,y2(i,j))/y2(i,j);
        
    end
    accuracy2(j) = accuracy2(j)/nB(2);
end
 
hold off
plot(accuracy1)
hold on 
plot(accuracy2)
title('Accuracy of gradient descent on logistic loss function with Lambda regularized objective')
legend('Accuracy of 0','Accuracy if not 0')
xlabel('Number of iteration')
ylabel('Accuracy')
    

%% 1.4 Accelerated Gradient descent on logistic loss function with Lambda regularized objective

% variable [h_i c]' for i=1, ... , 28^2

nIter = 20;
lambda = 0.000001;
x= zeros(shape(1)*shape(2)+1,1);
xOut = zeros(shape(1)*shape(2)+1,nIter);
L = 10

xOut = accGradient(x,nIter,L,lambda,data0,data1);



nA = size(data0);
nB = size(data1);

y = zeros(nA(2),nIter+1);
y2 = zeros(nB(2),nIter+1);
accuracy1 = zeros(nIter+1,1);
accuracy2 = zeros(nIter+1,2);

for j = 1:nIter+1
    
    for i = 1:nA(2)
        y(i,j) = xOut(1:end-1,j)'*data0(:,i) + xOut(end,j);
        
        accuracy1(j) = accuracy1(j) + max(0,y(i,j))/y(i,j);
        
    end
    
    accuracy1(j) = accuracy1(j)/nA(2) ;
    
    for i = 1:nB(2)
        y2(i,j) = xOut(1:end-1,j)'*data1(:,i) + xOut(end,j) ; 
        
        accuracy2(j) = accuracy2(j) + min(0,y2(i,j))/y2(i,j);
        
    end
    accuracy2(j) = accuracy2(j)/nB(2);
end
 
hold off
plot(accuracy1)
hold on 
plot(accuracy2)
title('Accuracy of accelerated gradient descent on logistic loss function with Lambda regularized objective')
legend('Accuracy of 0','Accuracy if not 0')
xlabel('Number of iteration')
ylabel('Accuracy')

