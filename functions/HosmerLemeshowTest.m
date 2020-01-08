function p = HosmerLemeshowTest(Yest,Ytrue,n)
  
    % This function implements the HosmerLemeshow goodness-of-fit test. When
    % the p-value is larger than a given significance level (e.g. 0.05), the
    % model is calibrated.
    %
    % Inputs:
    % Yest - 1D vector that contains model outputs. The values should be between 0 and 1.
    % Ytrue - 1D vector of binary target values. The values should be either 0 or 1.
    %
    % Outputs:
    % p - p-value
    %
    % Written by Joon Lee, August 2010.


    [Yest_sorted,idx]=sort(Yest);
    Ytrue_sorted=Ytrue(idx);

    decileSize=round(length(Yest)/n);
    HLstat=0;
    O=zeros(n,1);
    E=zeros(n,1);

    % first 9 bins
    for i=1:n-1
        first=(i-1)*decileSize+1;
        last=i*decileSize;
        O(i)=sum(Ytrue_sorted(first:last));
        E(i)=sum(Yest_sorted(first:last));
        HLstat=HLstat+(O(i)-E(i))^2/E(i)/(1-E(i)/decileSize);
    end

    % 10th bin (possibly with a different size than the other bins)
    first=(n-1)*decileSize+1;
    O(n)=sum(Ytrue_sorted(first:end));
    E(n)=sum(Yest_sorted(first:end));
    nn=length(Ytrue_sorted(first:end));
    HLstat=HLstat+(O(n)-E(n))^2/E(n)/(1-E(n)/nn);

    p=1-cdf('chi2',HLstat,n-2);