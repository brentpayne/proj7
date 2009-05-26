function [  ] = multinomial(  )
%UNTITLED1 Summary of this function goes here
%   Detailed explanation goes here
%logClassic400 = sparse(log(data.classic400+1));

num_xval = 10;

sample_num =20;

%data = load('classic400.mat');
labels = csvread('labelfile3');
matrix = csvread('cvs4matlab3');
wordlist = textread('wordlist3','%[^\n]');
labels = labels;
remove = (sum(matrix)==1);

matrix(:,remove) = [];

wordlist(remove,:) = [];



num_docs = size(matrix,1);
size_vocab = size(matrix,2);

min_label = min(labels);
max_label = max(labels);

topic1 = matrix(labels==1,:);
topic2 = matrix(labels==2,:);

size(topic1)
size(topic2)

%print_lsa(matrix, b, wordlist);
b = ones(size(topic1,1),1);
lsa1 = topic1\b;
%size(topic1)
%size(lsa1)
%b-(topic1*lsa1)
t1 = top_words(lsa1,sample_num,wordlist);

%print_lsa(matrix, b, wordlist);
lsa2 = topic2\ones(size(topic2,1),1);
t2 = top_words(lsa2,sample_num,wordlist);


t12 = top_words(abs(lsa1-lsa2),2*sample_num,wordlist);

%b=zeros(size(labels));
%b(labels==1)=1;
%b(labels~=1)=-1;
%lsaM1 = matrix\b;
%t3 = top_words(lsaM1,sample_num,wordlist);

%b(labels==2)=1;
%b(labels~=2)=-1;
%lsaM2 = matrix\b;
%t4 = top_words(lsaM2,sample_num,wordlist);

%t34 = top_words(abs(lsaM1-lsaM2),2*sample_num,wordlist);

%t1234 = top_words(abs(lsa1-lsa2+lsaM1-lsaM2),2*sample_num,wordlist);

tr = [1:size(matrix,2)];
%tt = [t1' t2' t3' t4'];
%tt = [t1' t2'];
%tt = [t3' t4'];
tt = [t12'];
%tt = [ t34'];
%tt = [ t1234' ];
tr(:,tt)=[];
matrix(:,tr)=[];
wordlist(tr,:)=[];

size(matrix)

topic1 = matrix(labels==1,:);
shuffled_input1 = sortrows( [randn(size(topic1,1), 1) topic1] );
shuffled_input1 = shuffled_input1(:,2:end);
%shuffled_input1 = topic1;
topic2 = matrix(labels==2,:);
shuffled_input2 = sortrows( [randn(size(topic2,1), 1) topic2] );
shuffled_input2 = shuffled_input2(:,2:end);
%shuffled_input2 = topic2;

topic_set = {shuffled_input1; shuffled_input2};

total=[0 0; 0 0];
twl = {['START'];['STARTR']};
totacc=0.0;
%X-Validation
for step=1:num_xval
    dataset = { {}, {} };
    for t=min_label:max_label
        [train, test] = create_traintest(step,num_xval,topic_set{t});
        if(num_xval == 1)
            train = test;
        end
        dataset{t} = {train,test};
    end
    test_set = [dataset{1}{2};dataset{2}{2}];
    test_labels = [ ones(size(dataset{1}{2},1),1) ;
        ones(size(dataset{2}{2},1),1) .*2  ];
    py= [ size(dataset{1}{1},1);  
        size(dataset{2}{1},1)] / sum( [size(dataset{1}{1},1),size(dataset{2}{1},1)] );
    log_theta = [ calculate_thetas(dataset{1}{1}), calculate_thetas(dataset{2}{1}) ];
    pred_x = classify_px( py, log_theta, test_set, test_labels );
    
    results = 2.^test_labels.*3.^pred_x;
    err0=sum(results==0);
    err1=sum(results<6);
    tp1=sum(results==6);
    fp12=sum(results==3*3*2);
    tp2=sum(results==3*3*2*2);
    fp21=sum(results==3*2*2);

    confusion_matrix= [tp1,fp12;fp21,tp2];
    acc = (tp1+tp2)/size(results,1);
    totacc = totacc + acc;

    total = total + confusion_matrix;
    trival = size(topic1,1)/size(matrix,1);
    avgacc = totacc / step;

    for t = [min_label:max_label]
        top_five = get_top_n_rows(log_theta(:,t), 7);
        display('topic');
        wl = [];
        for w = top_five(:,1)'
            w;
            wl = [wl, ' ', wordlist{w}];
        end
        twl{t,1} = [twl{t,1} wl];
    end
    
    nothing=0;

end
   
total
avgacc = totacc / step
twl{1,1}
twl{2,1}
    
function[] = print_lsa(A, b, words)
    size(A);
    size(b);

    lsa = [(1:size(A,2))', A\b];
    %lsa = sparse([(1:size(topic,2))', topic\ones(size(topic,1),5)]);
    %size(lsa)
    %rank1_filter = lsa( lsa(:,2) == 1, : )
    rank1_filter = (sortrows(lsa,(2)));
    rank1_filter = (sortrows(lsa,(-2)));
    words{rank1_filter(1:5,1)};
       
    
    


function[train, test] = create_traintest(step_num, total_steps, data)
%order = [1:size(data,1)]';
%ordered_data = [order data];
if(total_steps == 1)
    train = data;
    test = data;
else
    step_size = floor(size(data,1)/total_steps);
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
%v = log(x+1);
v=x;
function [v] = mylog2(x)
%v = mylog(x);
v=x;

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
    

function[word_nums] = top_words(m,n, wordlist)
t = get_top_n_rows(m, n);
wl = [];
for w = t(:,1)'
    w;
    wl = [wl, ' ', wordlist{w}];
end
wl
word_nums=t(:,1);

    
function[toprows] = get_top_n_rows(m, n)
cm = [ [1:size(m,1)]' m ];
sorted = sortrows(cm, [-2]);
toprows = sorted(1:(n),:);
    
%for doc=single_topic_matrix
