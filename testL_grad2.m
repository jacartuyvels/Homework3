clear all
hold off
data = load('Homework-3-data.mat');

training = data.training;
test = data.test;

nIter = 10


label = training.labels;
image = training.images;
shape = size(image);

image2 = test.images;
label2 = test.labels;
shape2 = size(image2);

image = reshape(image,[shape(1)*shape(2), shape(3)]);
[data0 data1] = partition(image,label,0);


image2 = reshape(image2,[shape2(1)*shape2(2), shape2(3)]);
[data20 data21] = partition(image2,label2,9);




data0 = data0(:,1:800);
data1 = data1(:,1:4800);
nA = size(data0,2);
nB = size(data1,2);

L = [5 10 30 60];

x= zeros(shape(1)*shape(2)+1,1);
xOut = zeros(shape(1)*shape(2)+1,nIter);

lambda = 10;
objective = zeros(nIter,length(L));
normh = zeros(nIter,length(L));
for k =1:length(L)
   
    xOut = accGradient(x,nIter,lambda,L(k),data0,data1);

    y = zeros(nA,nIter);
    y2 = zeros(nB,nIter);

    for j = 1:nIter
        y(:,j) = xOut(1:end-1,j)'*data0(:,:) +xOut(end,j);
        y2(:,j) = xOut(1:end-1,j)'*data1(:,:) +xOut(end,j);
        
        normh(j,k) = norm(xOut(1:end-1,j));

        objective(j,k) = objective(j,k) + sum(log(1+exp(-y(:,j))))/nA + sum(log(1+exp(y2(:,j))))/nB + lambda/2*normh(j,k);
    end
end

%%
figure

x= linspace(1,nIter,nIter);
plot(x,objective(:,1),x,objective(:,2),x,objective(:,3),x,objective(:,4))
legend('L=5','L=10','L=30','L=50')
xlim([1 nIter])
xlabel('Number of iterations')



