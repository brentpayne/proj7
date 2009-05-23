function [  ] = multinomial(  )
%UNTITLED1 Summary of this function goes here
%   Detailed explanation goes here
data = load('classic400.mat');
logClassic400 = sparse(log(data.classic400+1));
num_docs = size(data.classic400,1);
size_vocab = size(data.classic400,2);
topic1 = data.classic400(data.truelabels==1,:);
topic2 = data.classic400(data.truelabels==2,:);
topic3 = data.classic400(data.truelabels==3,:);
py= [ sum(data.truelabels==1);  
    sum(data.truelabels==2);  
    sum(data.truelabels==3)]/num_docs ;
log_theta = [ calculate_thetas(topic1), calculate_thetas(topic2), calculate_thetas(topic3) ];
pred_x = classify_px(py, log_theta, data.classic400, data.truelabels);
size(pred_x)
size(data.truelabels)
results = 2.^data.truelabels.*3.^pred_x';
err0=sum(results==0);
err1=sum(results<6);
tp1=sum(results==6);
fp12=sum(results==3*3*2);
fp13=sum(results==3*3*3*2);
tp2=sum(results==3*3*2*2);
fp21=sum(results==3*2*2);
fp23=sum(results==3*3*3*2*2);
fp31=sum(results==3*2*2*2);
fp32=sum(results==3*3*2*2*2);
tp3=sum(results==3*3*3*2*2*2);

[tp1,fp12,fp13;
 fp21,tp2,fp23;
 fp31,fp32,tp3]

function [pred_x] = classify_px(py, log_theta, M, labels)
minlabel = min(labels);
maxlabel = max(labels);
numlabels = maxlabel-minlabel+1;
pred_x= zeros(size(M,1),1);
for n = 1:size(M,1)
    tmp = -Inf;
    for t = 1:numlabels
        log_theta_x = log(M(n,:)+1)*log_theta(:,t);
        %log_multi= log_multinomial_const(M(n,:));
        pred_val = log(py(t))+log_theta_x;
        if(pred_val>tmp)
            pred_x(n,1) = t;
            tmp = pred_val;
        end
    end
end

function[v]=log_multinomial_const(x)
n = sum(x);
sum_log_factorial_x=0;
for w = x
    sum_log_factorial_x = sum_log_factorial_x + log_factorial(w);
end

v = log_factorial(n) - sum_log_factorial_x;


function[v] =log_factorial(n)
v=0;
if n~=0
    v = (n*(n+1))/2.0;
end


function[ smoothing_consts ] = smoothing(doc, M)
smoothing_consts = ones(size(doc))*1;

function [ thetas ] = calculate_thetas(single_topic_matrix)
thetas = zeros(size(single_topic_matrix,2),1);
running_total=0;
for doc = single_topic_matrix';
    doc = log(doc + 1 + smoothing(doc,single_topic_matrix)); % +1 for log shift, +1 for delta smoothing
    thetas = thetas + doc;
    running_total = running_total + sum(doc);
end
thetas = log(thetas)-log(running_total);
    
    
    

%for doc=single_topic_matrix