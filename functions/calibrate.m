function [perf, X, Y] = calibrate(actual,predicted,trueClass,Nbins)
%%
% perf = CALIBRATE(actual,predicted,trueClass,Nbins) 
%
% Returns performance metrics of your model.
%
%   actual    : Ground Truth (binary class labels, 1 or 0)
%   predicted : Model predictions
%   trueClass : Label for Class of Interest (1 or 0)
%   Nbins     : Number of bins to use for HL-test
%
% AUTHOR: MOHAMMAD GHASSEMI ~ March 2016
% EDITED: Tuka Alhanai ~ March 2016

%% SWITCH OFF WARNING
warning('off','MATLAB:nargchk:deprecated');

%% FLAGS FOR SECTIONS
calFlag = 0;
bbqFlag = 0;


%% MAKING SURE EVERYTHING IS A COLUMN VECTOR (BBQ cries otherwise)
if isrow(predicted)
    predicted = predicted';
end

if isrow(actual)
    actual = actual';
end

%% SCALING PREDICTIONS
% predicted_sc = (predicted - min(predicted)) / max((predicted - min(predicted)));
predicted_sc = predicted / max(predicted);
% predicted_sc = predicted;

%% CALCULATE AUC
% AUC - AREA UNDER THE RECEIVER OPERATING CURVE
[X,Y,T,AUC,OPTROCPT] = perfcurve(actual, predicted,trueClass);
% if AUC < 0.5
%     [X,Y,T,AUC,OPTROCPT] = perfcurve(actual, predicted,~trueClass);
% end

% plot(X,Y)
% hold on
% plot(OPTROCPT(1),OPTROCPT(2),'rx')
    
 
%% ROUNDING INTO PERCENTILES
Xr = round(X*100)/100;
Yr = round(Y*100)/100;
 
%plot(Xr,Yr);
%xlabel('False Predicted Dead'); ylabel('True Predicted Survivors')
 
%% EXTRACT CALIBRATION
if calFlag == 1
    step = 0.10; indi = 1;
    this = round(10*(predicted))/(10);
    range= 0:step:1;
    clear percy;
    for iii =[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        percy(indi)= nanmean(actual(find(this == iii)));
        indi = indi+1;
    end
end 

%% BBQ CALIBRATION
% DOWNLOAD TOOLKIT FROM https://github.com/pakdaman/calibration
if bbqFlag == 1
    addpath calibration-master/BBQ/
    options.N0 = 2;
    BBQ = build(predicted, actual, options);
    predicted2 = predict(BBQ, predicted', 1);

    step = 0.10; indi = 1;
    this = round(10*(predicted2))/(10);
    range= 0:step:1;
    clear percyBBQ;
    for iii =[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        percyBBQ(indi)= nanmean(actual(find(this == iii)));
        indi = indi+1;
    end
    rmpath calibration-master/BBQ/
end
 
%% EXTRACT ERROR RATE

% ACCURACY AT OPTIMUM ROC POINT
ind_opt = find(X == OPTROCPT(1) & Y == OPTROCPT(2));
OPT_ER = nanmean(((predicted > T(ind_opt(1))) == actual));

% THE EER POINT IS THE ACCURACY WHERE THE DIAGONAL LINE CROSS THE ROC LINE
% get line intercept code from: 
% http://www.mathworks.com/matlabcentral/fileexchange/22441-curve-intersections/content/InterX.m
P = InterX([X,Y]',[[0:0.01:1];[1:-0.01:0]]);
ind_opt = find(X == OPTROCPT(1) & Y == OPTROCPT(2));
EER = nanmean(((predicted > T(ind_opt(1))) == actual));
 
%% EXTRACT ALL OTHER METRICS
ind = 1;

perf(ind).AUC                   = AUC;
perf(ind).OPT_ER                = OPT_ER;
perf(ind).EER                   = EER;

% THIS ASSUMES THE PREDICTED PROBABILITIES ARE BETWEEN 0 and 1
% IF DECISION MAKING IS AT 0.5 THRESHOLD, THEN GET ACCURACY, etc.
% UNLIKE THE AUC MEASURE - THE FOLLOWING METRICS ASSUME THAT ALL 
% MISCLASSIFCICATIONS ARE EQUAL
perf(ind).Accuracy              = nanmean(round(predicted_sc) == actual);

% TRUE POSITIVE, etc. (GOOD TO USE FOR CONFUSION MATRIX)
perf(ind).TP                    = sum((round(predicted_sc) == trueClass) & (actual == trueClass));
perf(ind).FP                    = sum((round(predicted_sc) == trueClass) & (actual ~= trueClass));
perf(ind).TN                    = sum((round(predicted_sc) ~= trueClass) & (actual ~= trueClass));
perf(ind).FN                    = sum((round(predicted_sc) ~= trueClass) & (actual == trueClass));

% TP RATE, FP RATE, RECALL, PRECISION, F1 etc.
% FALL-OUT
perf(ind).FPR                   = perf(ind).FP / sum(actual ~= trueClass); 
% SENSITIVITY/RECALL
perf(ind).TPR                   = perf(ind).TP / sum(actual == trueClass);
perf(ind).Recall                = perf(ind).TPR;
perf(ind).Precision             = perf(ind).TP / sum(round(predicted_sc) == trueClass);
perf(ind).F1_score              = 2 * (perf(ind).Recall * perf(ind).Precision) / (perf(ind).Recall + perf(ind).Precision );

% PERFROMANCE AT FPR THRESHOLDS
perf(ind).TPR_when_FPR0p10      = Yr(max(find(Xr <= 0.10)));
perf(ind).TPR_when_FPR0p05      = Yr(max(find(Xr <= 0.05)));
perf(ind).TPR_when_FPR0         = Yr(max(find(Xr == 0)));

% PERFROMANCE AT TPR THRESHOLDS
perf(ind).FPR_when_TPR0p90      = Xr(min(find(Yr >= 0.90)));
perf(ind).FPR_when_TPR0p95      = Xr(min(find(Yr >= 0.95)));
perf(ind).FPR_when_TPR1p00      = Xr(min(find(Yr == 1)));

% CALLIBRATION
perf(ind).HL_test               = HosmerLemeshowTest(predicted,actual,Nbins);
% perf(ind).calibration           = percy;
% perf(ind).calibration_mSSE      = nanmean((percy - [0:.1:1]).^2);
% perf(ind).calibration_SSE       = nansum((percy - [0:.1:1]).^2);
% perf(ind).calibrationBBQ        = percyBBQ;
% perf(ind).calibrationBBQ_mSSE   = nanmean((percyBBQ - [0:.1:1]).^2);
% perf(ind).calibrationBBQ_SSE    = nansum((percyBBQ - [0:.1:1]).^2);
 
%% EOF