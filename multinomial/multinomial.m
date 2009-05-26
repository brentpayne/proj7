function [  ] = multinomial(  )
%UNTITLED1 Summary of this function goes here
%   Detailed explanation goes here
data = load('classic400.mat');
%logClassic400 = sparse(log(data.classic400+1));
num_docs = size(data.classic400,1);
size_vocab = size(data.classic400,2);

num_xval = 10;

min_label = min(data.truelabels);
max_label = max(data.truelabels);

topic1 = data.classic400(data.truelabels==1,:);
shuffled_input1 = sortrows( [randn(size(topic1,1), 1) topic1] );
shuffled_input1 = shuffled_input1(:,2:end);
topic2 = data.classic400(data.truelabels==2,:);
shuffled_input2 = sortrows( [randn(size(topic2,1), 1) topic2] );
shuffled_input2 = shuffled_input2(:,2:end);
topic3 = data.classic400(data.truelabels==3,:);
shuffled_input3 = sortrows( [randn(size(topic3,1), 1) topic3] );
shuffled_input3 = shuffled_input3(:,2:end);

b = (data.truelabels==1)'
print_lsa(data.classic400, b, data.classicwordlist);
rank1 = topic1\ones(size(topic1,1),1);
rank1
 
topic_set = {shuffled_input1; shuffled_input2; shuffled_input3};

%X-Validation
for step=1:num_xval
    dataset = { {}, {}, {} }
    for t=min_label:max_label
        [train, test] = create_traintest(step,num_xval,topic_set{t});
        dataset{t} = {train,test};
    end
    test_set = [dataset{1}{2};dataset{2}{2};dataset{3}{2}];
    test_labels = [ ones(size(dataset{1}{2},1),1) ;
        ones(size(dataset{2}{2},1),1) .*2 ;
        ones(size(dataset{3}{2},1),1) .*3 ];
    py= [ size(dataset{1}{1},1);  
        size(dataset{2}{1},1);  
        size(dataset{3}{1},1)] / sum( [size(dataset{1}{1},1),size(dataset{2}{1},1),size(dataset{3}{1},1)] );
    log_theta = [ calculate_thetas(dataset{1}{1}), calculate_thetas(dataset{2}{1}), calculate_thetas(dataset{3}{1}) ];
    pred_x = classify_px( py, log_theta, test_set, test_labels );
    
    results = 2.^test_labels.*3.^pred_x;
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
    for t = [min_label:max_label]
        topic_theta = [log_theta(:,t) [1:size(log_theta,1)]'];
        top_five = (sortrows(topic_theta,[-1]));
        top_five = top_five(1:5,2);
        [data.classicwordlist{top_five}]
    end
    nothing=0;

end

function[] = print_lsa(A, b, words)
    %size(A)
    %size(b)

    lsa = [(1:size(A,2))', A\b];
    %lsa = sparse([(1:size(topic,2))', topic\ones(size(topic,1),5)]);
    %size(lsa)
    %rank1_filter = lsa( lsa(:,2) == 1, : )
    rank1_filter = (sortrows(lsa,(2)))
    rank1_filter = (sortrows(lsa,(-2)))
    words{rank1_filter(1:5,1)}
       
    
    


function[train, test] = create_traintest(step_num, total_steps, data)
%order = [1:size(data,1)]';
%ordered_data = [order data];
step_size = size(data,1)/total_steps;
start = step_size*(step_num-1)+1;
stop = step_size*step_num;
if(step_num == 1)
    train = data(stop+1:end,:);
    test = data(1:stop,:);
elseif(step_num == total_steps)
    train = data(1:start-1,:);
    test = data(start:end,:);
else
    train = [data(1:start-1,:);data(stop+1:end,:)];
    test = data(start:stop,:);
end    


function [pred_x] = classify_px(py, log_theta, M, labels)
minlabel = min(labels);
maxlabel = max(labels);
numlabels = maxlabel-minlabel+1;
pred_x= zeros(size(M,1),1);
for n = 1:size(M,1)
    tmp = -Inf;
    for t = 1:numlabels
        log_theta_x = mylog(M(n,:))*log_theta(:,t);
        %log_multi= log_multinomial_const(M(n,:));
        pred_val = log(py(t))+log_theta_x;
        if(pred_val>tmp)
            pred_x(n,1) = t;
            tmp = pred_val;
        end
    end
end

function [v] = mylog(x)
v = log(x+1);
%v=x;
function [v] = mylog2(x)
v = mylog(x);
%v=x;

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
smoothing_consts = ones(size(doc))*(1.0/size(M,2));

function [ thetas ] = calculate_thetas(single_topic_matrix)
thetas = zeros(size(single_topic_matrix,2),1);
running_total=0;
for doc = single_topic_matrix';
    doc = mylog2(doc + smoothing(doc,single_topic_matrix)); % +1 for log shift, +1 for delta smoothing
    thetas = thetas + doc;
    running_total = running_total + sum(doc);
end
thetas = log(thetas)-log(running_total);
    
    
    

%for doc=single_topic_matrix
