clear all
hold off
data = load('Homework-3-data.mat');

training = data.training;
test = data.test;

nIter = 500


label = training.labels;
image = training.images;
shape = size(image);



image = reshape(image,[shape(1)*shape(2), shape(3)]);
[data0 data1] = partition(image,label,0);

nA = size(data0,2);
nB = size(data1,2);
data0 = data0(:,1:800);
data1 = data1(:,1:4800);

R = 8;

x= zeros(shape(1)*shape(2)+1,1);
xOut = zeros(shape(1)*shape(2)+1,nIter);
L = 60;

objective = zeros(nIter,length(L));


xOut = gradient1(x,nIter,R,L,data0,data1);

nA = size(data0);
nB = size(data1);
k=1

y
for j = 1:nIter
    
    for i = 1:nA(2)
        y1 = xOut(1:end-1,j)'*data0(:,i) +xOut(end,j);
        objective(j,k) = objective(j,k) + log(1+exp(-y1))/nA(2);
    end

    %accuracy21(j,k) = accuracy21(j,k)/nA(2) ;
    
    for i = 1:nB(2)
        y2 = xOut(1:end-1,j)'*data1(:,i) + xOut(end,j);
        objective(j,k) = objective(j,k) + log(1+exp(y2))/nB(2);

    end
    %accuracy22(j,k) = accuracy22(j,k)/nB(2);
end


%%
% hold off 
% figure 
% x= linspace(1,500,500);
% subplot(1,2,1)
% plot(x,accuracy21(:,1),x,accuracy21(:,2),x,accuracy21(:,3),x,accuracy21(:,4),x,accuracy21(:,5))
% legend('L=1','L=5','L=10','L=20','L=50')
% ylim([0.8 1])
% xlim([0 500])
% xlabel('Number of iterations')
% subplot(1,2,2)
% plot(x,accuracy21(:,1),x,accuracy21(:,2),x,accuracy21(:,3),x,accuracy21(:,4),x,accuracy21(:,5))
% legend('L=1','L=5','L=10','L=20','L=50')
% ylim([0.95 0.99])
% xlim([0 500])
% xlabel('Number of iterations')
% figure 
% subplot(1,2,1)
% plot(x,accuracy22(:,1),x,accuracy22(:,2),x,accuracy22(:,3),x,accuracy22(:,4),x,accuracy22(:,5))
% legend('L=1','L=5','L=10','L=20','L=50')
% title('Accuracy pour != 0')
% ylim([0.8 1])
% xlim([0 500])
% xlabel('Number of iterations')
% subplot(1,2,2)   
% plot(x,accuracy22(:,1),x,accuracy22(:,2),x,accuracy22(:,3),x,accuracy22(:,4),x,accuracy22(:,5))
% legend('L=1','L=5','L=10','L=20','L=50')
% title('Accuracy pour != 0')
% ylim([0.95 0.99])
% xlim([0 500])
% xlabel('Number of iterations')


hold off 
figure 
x= linspace(1,nIter,nIter);
plot(x,objective(:,1))
legend('L=60')
xlim([0 500])
ylim([0 5])
xlabel('Number of iterations')
