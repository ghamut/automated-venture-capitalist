% Script contains modeling approach presented in AAAI-KDF '20 Paper
%   'The Automated Venture Capitalist: Data and Methods to Predict the
%    Fate of Startup Ventures'
%
%   Run the script from start to end to format data, perform
%   leave-one-team-out cross-validation modeling, extract logistic 
%   regression model coefficients, and conduct a cost analysis.
%
%   Input:  'data/individualData.csv'
%   Output: 'results/results.csv'
%           'modelCoeff/*csv'
%           'figures/*png'
%           'results/comparison.csv'
%
%
% Authors: Mohammad M. Ghassemi and Tuka Alhanai
% Created: August 2017
% Last Updated: January 7th 2020

%% Initialization
clear all;
clc;

% suppress some warnings to keep console clean
warning('off','stats:glmfit:IterationLimit')

% contains functions called by this script
addpath('functions/')

%% 0. Load in the 100K Data
disp(' ... 1. Loading data: individualData.txt')
% the dataset is a [people x features] matrix. We collected a
% total of 613 individuals in the original dataset, but here we
% have removed 266 inviduals with incomplete data, reducing the
% total number of teams from 192, to 177.

% To make the analysis easier, we rename the data table as 'd'.
d = readtable('data/individualData.csv');

%% 1. Select 17 features
disp(' ... 2. Computing features')
% We can select up to 17 features becasue we need to keep the data
% to features ratio at 100-1. 
s=[]; s.team_id = d.team_id;

% INDIVIDUAL-LEVEL FEATURES: 8 in total %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Harvard or MIT?
s.is_mit = d.is_mit;
s.is_harvard = d.is_harvard;

% How long since graduation?
s.years_since_graduation = d.years_since_graduation;

% Do you Hold an MBA and/or a PhD?
s.is_mba =1*(d.is_mba);
s.is_phd = d.is_phd;

% Did you study Math Science or Engineering?
s.is_stem =1*( d.major_engineer | d.major_scientist | d.major_mathecon);

% How many of the the 7 soft skills do you have?
s.skills_soft =(d.skill_class_creative + d.skill_class_design + d.skill_class_communication +  d.skill_class_legal + d.skill_class_management + d.skill_class_relationship + d.skill_class_education)/7;

% Do people consistently think you look competent?
s.pic_rating_total_3 = d.pic_rating_total == 3;

% TEAM LEVEL FEATURES: 8 Total  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Do at least 40% of people prefer your idea and team name to random other ideas?
s.idea_rating_2min = d.votes_threshold_2min/10;

% Was the idea something that would be used everyday, periodically, or rarely. 
s.topic_inessential = d.topic_food | d.topic_goods | d.topic_impact | d.topic_entertainment ;
s.topic_essentialperiodic  =  d.topic_employment | d.topic_information | d.topic_children | d.topic_social |  d.topic_health | d.topic_transportation ;
s.topic_essentialdaily  = d.topic_finance | d.topic_energy ;

% How much did the the author directly engage with the reader?
s.wd_speaking_to_you = d.wd_you./max(d.POS_period,1);
s.wd_questions = d.wd_questions./max(d.POS_period,1);

% Were they descriptive?
s.wd_adjectives = d.wd_adjective./max(d.POS_period,1);

% Were they positive?
s.positive_sentiment = d.sent_Verypositive./max(d.POS_period,1) + d.sent_Positive./max(d.POS_period,1);

% OUTCOMES: nominated, finalist, and who was successful %%%%%%%%%%%%%%%%%%%
s.nomination = d.nomination;
s.success = d.success;
s.finalist = d.finalist;

s = struct2table(s);
d = s;

%% 2. Condense features into the team-level. 
disp(' ... 3. Condensing features into team-level')
% we will take an average of the features available within
% each of the team variables.

%get the unique teams.
t = [];
team_ids = unique(d.team_id);
for i = 1:length(team_ids)
    %collect all the people that are members of this team.
    this_team_id = team_ids(i);
    these_members = find(d.team_id == this_team_id);

    %If there is more than one member on the team.
    if length(these_members) > 1
        t = [t; mean(d{these_members,:})];
    else
        t = [t; d{these_members,:}];
    end
end
t = array2table(t,'VariableNames',d.Properties.VariableNames);
clearvars -except d t

% write to file
filename = 'data/teamData.csv';
fprintf('   ---> saving file %s \n', filename)
writetable(t,filename,'Delimiter',',');  

% read from file
fprintf('   <--- reading file %s \n', filename)
t = readtable(filename);
t.team_id = [];

%% 3. Univariate Correlations Analysis (Individual Level)
disp(' ... 4a. Performing uni-variate correlation analysis (individual-level)')
% In this first portion of the analysis, We computed
% The univariate correlations and p-values between the
% selected features and the following three outcomes:
% (1) nomination, (2) success and, (3) finalist status.

%A. Identify the location of the outcomes in the data matrix.
success_index   = find(ismember(d.Properties.VariableNames,'success'));
nominated_index = find(ismember(d.Properties.VariableNames,'nomination'));
finalist_index  = find(ismember(d.Properties.VariableNames,'finalist'));

%B. compute correlations:
[corr_mat p_mat] = corr(d{:,:});

%C. generate the correlation table.
corrtab.feature_name   = d.Properties.VariableNames';
corrtab.success_corr   = corr_mat(:,success_index);
corrtab.success_p      = p_mat(:,success_index);
corrtab.nominated_corr = corr_mat(:,nominated_index);
corrtab.nominated_p    = p_mat(:,nominated_index);
corrtab.finalist_corr  = corr_mat(:,finalist_index);
corrtab.finalist_p     = p_mat(:,finalist_index);

%D. OUTPUT:
corrtab_individual = struct2table(corrtab);
clearvars -except d t corrtab_individual

%% 3. Univariate Correlations Analysis (Team Level)
disp(' ... 4b. Performing uni-variate correlation analysis (team-level)')
% In this first portion of the analysis, We computed
% The univariate correlations and p-values between the
% selected features and the following three outcomes:
% (1) nomination, (2) success and, (3) finalist status.

%A. Identify the location of the outcomes in the data matrix.
success_index   = find(ismember(t.Properties.VariableNames,'success'));
nominated_index = find(ismember(t.Properties.VariableNames,'nomination'));
finalist_index  = find(ismember(t.Properties.VariableNames,'finalist'));

%B. compute correlations:
[corr_mat p_mat] = corr(t{:,:});

%C. generate the correlation table.
corrtab.feature_name = t.Properties.VariableNames';

corrtab.success_corr   = corr_mat(:,success_index);
corrtab.success_p      = p_mat(:,success_index);
corrtab.nominated_corr = corr_mat(:,nominated_index);
corrtab.nominated_p    = p_mat(:,nominated_index);
corrtab.finalist_corr  = corr_mat(:,finalist_index);
corrtab.finalist_p     = p_mat(:,finalist_index);

%D. OUTPUT:
corrtab_team = struct2table(corrtab);
clearvars -except d t corrtab_individual corrtab_team

%% 4. TRAIN THE LOGISTIC REGRESSION MODELS 
disp(' ... 5. Training logistic regression models')
% We computed 10 Logistic Regression Models including the various types
% of features to see how much information in contained in the features.

t.success_and_nomination = t.success & t.nomination;
num_folds = height(t);

for i = 1:num_folds
    
    if rem(i,10) == 0
        fprintf('   ... processing fold: %d/%d \n',i, num_folds)
    end
    
    test  = t(i,:);
    train = t; train(i,:) = [];
    
   formula_succ = ['success ~' ... 
              'is_mit' ... 
              '+ is_harvard' ...
              '+ years_since_graduation'...
              '+ is_mba' ...
              '+ is_phd' ...
              '+ is_stem' ...
              '+ skills_soft' ...
              '+ pic_rating_total_3' ...
              '+ idea_rating_2min' ...
              '+ topic_inessential' ...
              '+ topic_essentialperiodic' ...
              '+ topic_essentialdaily' ...
              '+ wd_speaking_to_you' ...
              '+ wd_questions' ...
              '+ wd_adjectives' ...
              '+ positive_sentiment'];

   formula_succ_given_nom = ['success ~' ... 
              'is_mit' ... 
              '+ is_harvard' ...
              '+ years_since_graduation'...
              '+ is_mba' ...
              '+ is_phd' ...
              '+ is_stem' ...
              '+ skills_soft' ...
              '+ pic_rating_total_3' ...
              '+ idea_rating_2min' ...
              '+ topic_inessential' ...
              '+ topic_essentialperiodic' ...
              '+ topic_essentialdaily' ...
              '+ wd_speaking_to_you' ...
              '+ wd_questions' ...
              '+ wd_adjectives' ...
              '+ positive_sentiment' ...
              '+ nomination'];            
     
   formula_nom = ['nomination ~' ... 
              'is_mit' ... 
              '+ is_harvard' ...
              '+ years_since_graduation'...
              '+ is_mba' ...
              '+ is_phd' ...
              '+ is_stem' ...
              '+ skills_soft' ...
              '+ pic_rating_total_3' ...
              '+ idea_rating_2min' ...
              '+ topic_inessential' ...
              '+ topic_essentialperiodic' ...
              '+ topic_essentialdaily' ...
              '+ wd_speaking_to_you' ...
              '+ wd_questions' ...
              '+ wd_adjectives' ...
              '+ positive_sentiment'];

   formula_nom_onlymember = ['nomination ~' ... 
              'is_mit' ... 
              '+ is_harvard' ...
              '+ years_since_graduation'...
              '+ is_mba' ...
              '+ is_phd' ...
              '+ is_stem' ...
              '+ skills_soft' ...
              '+ pic_rating_total_3'];          
          
   formula_nom_onlyidea = ['nomination ~ idea_rating_2min' ...
              '+ topic_inessential' ...
              '+ topic_essentialperiodic' ...
              '+ topic_essentialdaily' ...
              '+ wd_speaking_to_you' ...
              '+ wd_questions' ...
              '+ wd_adjectives' ...
              '+ positive_sentiment'];
 
   formula_nom_onlycrowd = ['nomination ~ pic_rating_total_3' ...
              '+ idea_rating_2min'];          
          
    formula_succ_and_nom = ['success_and_nomination ~' ... 
              'is_mit' ... 
              '+ is_harvard' ...
              '+ years_since_graduation'...
              '+ is_mba' ...
              '+ is_phd' ...
              '+ is_stem' ...
              '+ skills_soft' ...
              '+ pic_rating_total_3' ...
              '+ idea_rating_2min' ...
              '+ topic_inessential' ...
              '+ topic_essentialperiodic' ...
              '+ topic_essentialdaily' ...
              '+ wd_speaking_to_you' ...
              '+ wd_questions' ...
              '+ wd_adjectives' ...
              '+ positive_sentiment'];         

    formula_succ_onlymember = ['success ~' ... 
              'is_mit' ... 
              '+ is_harvard' ...
              '+ years_since_graduation'...
              '+ is_mba' ...
              '+ is_phd' ...
              '+ is_stem' ...
              '+ skills_soft' ...
              '+ pic_rating_total_3'];          

          
    formula_succ_onlyidea = ['success ~' ... 
              'idea_rating_2min' ...
              '+ topic_inessential' ...
              '+ topic_essentialperiodic' ...
              '+ topic_essentialdaily' ...
              '+ wd_speaking_to_you' ...
              '+ wd_questions' ...
              '+ wd_adjectives' ...
              '+ positive_sentiment'];
    
    formula_succ_onlycrowd = ['success ~' ... 
              ' pic_rating_total_3' ...
              '+ idea_rating_2min'];

    formula_succ_onlynomination = ['success ~' ... 
              ' nomination'];
          
          
    mdl_succ          = fitglm(train,formula_succ,'distr','binomial');
    predicted_succ(i) = predict(mdl_succ,test);
    
    mdl_succ_given_nom          = fitglm(train,formula_succ_given_nom,'distr','binomial');
    predicted_succ_given_nom(i) = predict(mdl_succ_given_nom,test); 

    mdl_nom_onlyidea          = fitglm(train,formula_nom_onlyidea,'distr','binomial');
    predicted_nom_onlyidea(i) = predict(mdl_nom_onlyidea,test); 
 
    mdl_nom          = fitglm(train,formula_nom,'distr','binomial');
    predicted_nom(i) = predict(mdl_nom,test);     

    mdl_nom_onlymember          = fitglm(train,formula_nom_onlymember,'distr','binomial');
    predicted_nom_onlymember(i) = predict(mdl_nom_onlymember,test); 
    
    mdl_nom_onlycrowd          = fitglm(train,formula_nom_onlycrowd,'distr','binomial');
    predicted_nom_onlycrowd(i) = predict(mdl_nom_onlycrowd,test);     

    mdl_succ_and_nom          = fitglm(train,formula_succ_and_nom,'distr','binomial');
    predicted_succ_and_nom(i) = predict(mdl_succ_and_nom,test);   

    mdl_succ_onlymember          = fitglm(train,formula_succ_onlymember,'distr','binomial');
    predicted_succ_onlymember(i) = predict(mdl_succ_onlymember,test);   
    
    mdl_succ_onlyidea          = fitglm(train,formula_succ_onlyidea,'distr','binomial');
    predicted_succ_onlyidea(i) = predict(mdl_succ_onlyidea,test);   
    
    mdl_succ_onlycrowd          = fitglm(train,formula_succ_onlycrowd,'distr','binomial');
    predicted_succ_onlycrowd(i) = predict(mdl_succ_onlycrowd,test); 
    
    mdl_succ_onlynomination          = fitglm(train,formula_succ_onlynomination,'distr','binomial');
    predicted_succ_onlynomination(i) = predict(mdl_succ_onlynomination,test);      
       
end

%% 5. EVALUATE THE LOGISTIC REGRESSION MODELS
% calculaing model performance with calibrate() function
disp(' ... 6. Evaluating logistic regression models')

clear results;

%Compute the Area Under Reciever Operator Curve
r = calibrate(t.success,predicted_succ_given_nom,1,10);
r = [r calibrate(t.success,predicted_succ,1,10)];
r = [r calibrate(t.success,predicted_succ_onlyidea,1,10)];
r = [r calibrate(t.success,predicted_succ_onlymember,1,10)];
r = [r calibrate(t.success,predicted_succ_onlycrowd,1,10)];
r = [r calibrate(t.nomination,predicted_nom,1,10)];
r = [r calibrate(t.nomination,predicted_nom_onlyidea,1,10)];
r = [r calibrate(t.nomination,predicted_nom_onlymember,1,10)];
r = [r calibrate(t.nomination,predicted_nom_onlycrowd,1,10)];
r = [r calibrate(t.success_and_nomination,predicted_succ_and_nom,1,10)];

results = struct2table(r);
%Save the names of the models

results.model_type = {'Logistic_Regression',
                    'Logistic_Regression',
                    'Logistic_Regression',
                    'Logistic_Regression',
                    'Logistic_Regression',
                    'Logistic_Regression',
                    'Logistic_Regression',
                    'Logistic_Regression',
                    'Logistic_Regression',
                    'Logistic_Regression'};

results.model_name = {'Success_given_idea_member_crowd_nomination',
                      'Success_given_idea_member_crowd',
                      'Success_given_idea_crowd',
                      'Success_given_member_crowd',
                      'Success_given_crowd',
                      'Nomination_given_idea_member_crowd',
                      'Nomiantion_given_idea_crowd',
                      'Nomination_given_member_crowd',
                      'Nomination_given_crowd'
                      'NominationandSuccess_given_idea_member_crowd'};

clearvars -except d t corrtab_individual corrtab_team results predicted_succ

%% 6. TRY OTHER CLASSIFICATION LEARNER MODELS
disp(' ... 7. Modeling with other classification learners')

y = t.success;
x =  [t.is_mit';
      t.is_harvard';
      t.years_since_graduation';
      t.is_mba';
      t.is_phd';
      t.is_stem';
      t.skills_soft';
      t.pic_rating_total_3';
      t.idea_rating_2min';
      t.topic_inessential';
      t.topic_essentialperiodic';
      t.topic_essentialdaily';
      t.wd_speaking_to_you';
      t.wd_questions';
      t.wd_adjectives';
      t.positive_sentiment';
      t.nomination']';     

variable_names = {'is_mit', ... 
              'is_harvard', ...
              'years_since_graduation',...
              'is_mba' ,...
              'is_phd', ...
              'is_stem', ...
              'skills_soft', ...
              'pic_rating_total_3', ...
              'idea_rating_2min', ...
              'topic_inessential', ...
              'topic_essentialperiodic', ...
              'topic_essentialdaily', ...
              'wd_speaking_to_you', ...
              'wd_questions', ...
              'wd_adjectives', ...
              'positive_sentiment', ...
              'nomination'}';  

mdls = {'linearSVM', 'cubicSVM', 'quadraticSVM', 'mediumGaussianSVM', ...
'coarseGaussianSVM', 'linearDisc', 'quadraticDisc', 'simpleTree', ...
 'mediumTree', 'complexTree', 'RUSBoostedTrees', 'ensembleBoostedTrees', ...
'ensembleBaggedTrees', 'fineKNN', 'mediumKNN', 'coarseKNN', 'cosineKNN',...
'cubicKNN', 'weightedKNN', 'logReg'};          

% Train all the models...
for j = 1:length(mdls)
    clear predictions;
    
    fprintf('   ... Modeling: %s \n', mdls{j} )
    
    num_folds = height(t);
    for i = 1:num_folds
        
        if rem(i,50) == 0
            fprintf('      ... processing fold: %d/%d \n',i,num_folds)
        end

        x_test = x(i,:);
        y_test = y(i);
        x_train = x; y_train = y;
        x_train(i,:) = []; y_train(i,:) = [];

        [~, score(i,:), predictions(i)]=...
            trainClassifier(x_train,y_train,x_test,y_test,variable_names,mdls{j});  
    end
    r = calibrate(t.success,score(:,2),1,10);
    r.model_type = mdls{j};
    r.model_name ='Success_given_idea_member';
    r = struct2table(r);
    results = [results; r];
end

clearvars -except d t corrtab_individual corrtab_team results predicted_succ

% saving results to file
filename = 'results/results.csv';
fprintf('   ---> saving file %s \n', filename);
writetable(results,filename,'Delimiter',',')

%% 7. Save the best performing Nomination and Success Models
% - Output in 'modelCoeff/*csv' corresponds to Table 3 in paper.
% - Cost Analysis corresponds to figure 2 in paper.
disp(' ... 8. Calculating model coefficients and cost thresholds of best performing models')

formula = ['success ~' ... 
          'is_mit' ... 
          '+ is_harvard' ...
          '+ is_mba'...
          '+ years_since_graduation'...
          '+ is_phd' ...
          '+ is_stem' ...
          '+ skills_soft' ...
          '+ pic_rating_total_3' ...
          '+ idea_rating_2min' ...
          '+ topic_inessential' ...
          '+ topic_essentialperiodic' ...
          '+ topic_essentialdaily' ...
          '+ wd_speaking_to_you' ...
          '+ wd_questions' ...
          '+ wd_adjectives' ...
          '+ positive_sentiment'];

mdl_succ                   = fitglm(t,formula,'distr','binomial');
prediction                 = predict(mdl_succ,t);
t.best_success_predictions = prediction;
Success_Model_performance  = calibrate(t.success,prediction,1,10);
odds_ratio_95ci            = exp(mdl_succ.coefCI);
mdl_succ                   = mdl_succ.Coefficients;
mdl_succ.OddsRatio         = exp(mdl_succ.Estimate);
mdl_succ.OddsRatio_95CI    = odds_ratio_95ci;
Success_Model              = mdl_succ;

close
false_positive = [0.1 0.5 1 10];
for i = 1:4
    fixed_cost = 0;
    cost(1,1)  = 0;    % Cost of True  Negatives 
    cost(2,1)  = 0;    % Cost of False Negatives 
    cost(1,2)  = false_positive(i);   % Cost of False Positives
    cost(2,2)  = -1;   % Cost of True  Positives
    [X,Y,optimal_threshold_success,optimal_cost] = findoptimalclassifcationthreshold(cost,fixed_cost,prediction,t.success);
end

xlim([0 0.8])
ylim([-50 100])
title('Figure 2: Cost Plot of Survival')
xlabel('Optimal Classification Threshold')
ylabel('Cost')

filename = 'figures/costPlot_success.png';
fprintf('   ---> saving file %s \n', filename);
saveas(gcf, filename);

filename = 'modelCoeff/successModel_coeff.csv';
fprintf('   ---> saving file %s \n', filename)
writetable(Success_Model,filename,'Delimiter',',','WriteRowNames',true)

filename = 'modelCoeff/successModel_perf.csv';
fprintf('   ---> saving file %s \n', filename)
writetable(struct2table(Success_Model_performance),filename,'Delimiter',',')

formula = ['nomination ~' ... 
          'is_mit' ... 
          '+ is_harvard' ...
          '+ is_mba'...
          '+ years_since_graduation'...
          '+ is_phd' ...
          '+ is_stem' ...
          '+ skills_soft' ...
          '+ pic_rating_total_3' ...
          '+ idea_rating_2min' ...
          '+ topic_inessential' ...
          '+ topic_essentialperiodic' ...
          '+ topic_essentialdaily' ...
          '+ wd_speaking_to_you' ...
          '+ wd_questions' ...
          '+ wd_adjectives' ...
          '+ positive_sentiment'];

% model nomination outcome on all data
mdl_nom                       = fitglm(t,formula,'distr','binomial');
prediction                    = predict(mdl_nom,t);
t.best_nomination_predictions = prediction;
Nomination_Model_performance  = calibrate(t.nomination,prediction,1,10);
odds_ratio_95ci               = exp(mdl_nom.coefCI);
mdl_nom                       = mdl_nom.Coefficients;
mdl_nom.OddsRatio             = exp(mdl_nom.Estimate);
mdl_nom.OddsRatio_95CI        = odds_ratio_95ci;
Nomination_Model              = mdl_nom;

close
false_positive = [0.1 0.5 1 10];
for i = 1:4
    fixed_cost = 0;
    cost(1,1)  = 0;     % Cost of True  Negatives 
    cost(2,1)  = 0;     % Cost of False Negatives 
    cost(1,2)  = false_positive(i);    % Cost of False Positives
    cost(2,2)  = -1;    % Cost of True  Positives
    [~,~,optimal_threshold_nomination,optimal_cost] = findoptimalclassifcationthreshold(cost,fixed_cost,prediction,t.nomination);
end

xlim([0 1])
ylim([-50 100])
title('Cost Plot of Nomination')
xlabel('Optimal Classification Threshold')
ylabel('Cost')

filename = 'figures/costPlot_nomination.png';
fprintf('   ---> saving file %s \n', filename);
saveas(gcf,filename);

filename = 'modelCoeff/nominationModel_coeff.csv';
fprintf('   ---> saving file %s \n', filename);
writetable(Nomination_Model,filename,'Delimiter',',','WriteRowNames',true)

filename = 'modelCoeff/nominationModel_perf.csv';
fprintf('   ---> saving file %s \n', filename);
writetable(struct2table(Nomination_Model_performance),filename,'Delimiter',',')

clearvars -except d t corrtab_individual corrtab_team results Nomination_Model Success_Model Nomination_Model_performance Success_Model_performance optimal_threshold_nomination optimal_threshold_success predicted_succ     

%% PLOT THE % CORRECT BY THE PREDICTED PROBABILITIES:
% This visualizes model calibration (corrsponds to figure 1 in paper)
disp(' ... 9. Plotting calibration')

close

for i = 0:9
    rows = find(t.best_success_predictions > i/10  & t.best_success_predictions < i+.0999);
    actual_probability(i+1) = nanmean(t.success(rows));   
end


plot(0.1:.1:1,actual_probability,'o');
hold on
plot([0 1],[0 1]);
title('Figure 1: Calibration Plot')
xlabel('Predicted Probability of Survival (%)')
ylabel('True Probability of Survival (%)')

filename = 'figures/calibrationPlot_success.png';
fprintf('   ---> saving file %s \n', filename);
saveas(gcf,filename)

%% Comparing Corwd vs. Judges vs. Algorithm Performance
% This analysis is displayed in Table 2 in the paper.
disp(' ... 10. Comparing performance between crowd, judges, and models')

% how many successful teams are in the data? -- 0.2768
comp.info = 'how many successful teams are in the data?';
comp.val  = sum(t.success) / length(t.success);
compList  = [comp];

% how many nominated teams succeeded according to MIT100K judges? -- 0.3889
comp.info = 'how many nominated teams succeeded according to MIT100K judges?';
comp.val  = sum(t.nomination & t.success) / sum(t.nomination);
compList  = [compList; comp];

% how many nominated teams succeeded according to MIT100K judges, excluding finalists? --  0.3617
comp.info = 'how many nominated teams succeeded according to MIT100K judges, excluding finalists?';
comp.val  = sum(t.nomination & t.success & ~t.finalist) / sum(t.nomination & ~t.finalist);
compList  = [compList; comp];

% how many successful teams did the mit100k judges catch? -- 0.4286
comp.info = 'how many successful teams did the mit100k judges catch?';
comp.val  = sum(t.nomination & t.success) / sum(t.success);
compList  = [compList; comp];

% how many successful teams did the mit100k judges catch, excluding finalists? -- 0.3778
comp.info = 'how many successful teams did the mit100k judges catch, excluding finalists?';
comp.val = sum(t.nomination & t.success & ~t.finalist) / sum(t.success & ~t.finalist);
compList  = [compList; comp];

% how many unnominated teams became successful? -- 0.2276
comp.info = 'how many unnominated teams became successful?';
comp.val = sum(~t.nomination & t.success) / sum(~t.nomination);
compList  = [compList; comp];

% how many successful teams did the algorithm find? --  0.4490
[m,ind]   = sort(predicted_succ,2,'descend');
comp.info = 'how many successful teams did the algorithm find?';
comp.val  = sum(t.success(ind(1:sum(t.nomination)))) / sum(t.success);
compList  = [compList; comp];

% how many successful teams did the algorithm find if only nominated 54? -- 0.4074
comp.info = 'how many successful teams did the algorithm find if only nominated 54?';
comp.val  = sum(t.success(ind(1:sum(t.nomination)))) / sum(t.nomination);
compList  = [compList; comp];

% toss out finalists, how many successful teams did the algorithm find? (given success) -- 0.4222
comp.info = 'toss out finalists, how many successful teams did the algorithm find? (given success)'
comp.val  = sum(t.success(ind(1:sum(t.nomination))) & ~t.finalist(ind(1:sum(t.nomination))) ) / sum(t.success & ~t.finalist);
compList  = [compList; comp];

% toss out finalists, how many successful teams did the algorithm find? (given nomination) -- 0.4043
comp.info = 'toss out finalists, how many successful teams did the algorithm find? (given nomination)';
comp.val  = sum(t.success(ind(1:sum(t.nomination))) & ~t.finalist(ind(1:sum(t.nomination))) ) / sum(t.nomination & ~t.finalist);
compList  = [compList; comp];

% CROWD IDEA RATING AS AUC.
calibrate(t.nomination,t.idea_rating_2min ,1,10); % AUC: 0.6096
calibrate(t.finalist,t.idea_rating_2min ,1,10);   % 0.5718
calibrate(t.success,t.idea_rating_2min ,1,10);    % 0.5908

% rank
[m,ind] = sort(t.idea_rating_2min,1,'descend');

% how many successful teams did the algorithm find? - 0.3673
comp.info = 'how many successful teams did the algorithm find?';
comp.val  = sum(t.success(ind(1:sum(t.nomination)))) / sum(t.success);
compList  = [compList; comp];

% how many successful teams did the algorithm find if only nominated 54? -- 0.3333
comp.info = 'how many successful teams did the algorithm find if only nominated 54?';
comp.val  = sum(t.success(ind(1:sum(t.nomination)))) / sum(t.nomination);
compList  = [compList; comp];

% toss out finalists, how many successful teams did the algorithm find? (given success) -- 0.3556
comp.info = 'toss out finalists, how many successful teams did the algorithm find? (given success)';
comp.val  = sum(t.success(ind(1:sum(t.nomination))) & ~t.finalist(ind(1:sum(t.nomination))) ) / sum(t.success & ~t.finalist);
compList  = [compList; comp];

% toss out finalists, how many successful teams did the algorithm find? (given nomination) -- 0.3404
comp.info = 'toss out finalists, how many successful teams did the algorithm find? (given nomination)';
comp.val  = sum(t.success(ind(1:sum(t.nomination))) & ~t.finalist(ind(1:sum(t.nomination))) ) / sum(t.nomination & ~t.finalist);
compList  = [compList; comp];

filename = 'results/comparison.csv';
fprintf('   ---> saving file %s \n', filename);
writetable(struct2table(compList),filename,'Delimiter',',')