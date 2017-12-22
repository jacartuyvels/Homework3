clear all
hold off
data = load('Homework-3-data.mat');

training = data.training;
test = data.test;

nIter = 500


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



x= zeros(shape(1)*shape(2)+1,1);
xOut = zeros(shape(1)*shape(2)+1,nIter);

L = [10 30];

x= zeros(shape(1)*shape(2)+1,1);
xOut = zeros(shape(1)*shape(2)+1,nIter);

lambda = [10^(-2) 10^(-4) 10^(-6) 10^(-8)];

gradOut = zeros(length(L),length(lambda),nIter)

for i = 1:length(L)
   for j= 1:length(lambda)
        [xOut gradOut(i,j,:)] = gradient2(x,nIter,lambda(j),L(i),data0,data1);
   end
end

x = 1:nIter
grad1 = gradOut(1,:,:);
grad1 = reshape(grad1,[length(lambda) nIter]);
grad2 = gradOut(2,:,:)
grad2 = reshape(grad2,[length(lambda) nIter]);

subplot(1,2,1)
semilogy(x,grad1(1,:),x,grad1(2,:),x,grad1(3,:),x,grad1(4,:))
title('L=10')
legend('lambda = 10^{-2}','lambda = 10^{-4}','lambda = 10^{-6}','lambda = 10^{-8}')

subplot(1,2,2)
semilogy(x,grad2(1,:),x,grad2(2,:),x,grad2(3,:),x,grad2(4,:))
title('L=30')
legend('lambda = 10^{-2}','lambda = 10^{-4}','lambda = 10^{-6}','lambda = 10^{-8}')








