%% Homework 3 Optimization run.m

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

%% 1.2 Gradient descent with logistic loss function 

% variable [h_i c] for i=1, ... , 28^2



tol = 0;
nIter = 60;
R = 8;
x= zeros(shape(1)*shape(2)+1,1);
xOut = zeros(shape(1)*shape(2)+1,nIter);

xOut = gradient1(x,nIter,R,data0,data1);



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
    
    

