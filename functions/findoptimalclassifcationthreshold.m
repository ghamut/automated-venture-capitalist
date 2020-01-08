function [X,Y,optimal_threshold,optimal_cost] = findoptimalclassifcationthreshold(cost,fixed_cost,predictions,labels)
    %% IDENTIFY THE OPTIMAL CLASSIFICATION THRESHOLD GIVEN A COST MATRIX:
    %
    % Cost analysis %%%%%%%%%%%%%%%%%%%%%%%%
    %
    %% AUTHOR: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %         Mohammad M. Ghassemi
    %         Wednesday, August 16th, 2017
    %% PURPOSE: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %         Identify an optimal classification threshold, given information
    %         About the costs of operating the algorithm, and misclassifcaiton.
    %% INPUTS(4): %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %1. cost: A 2x2 matrix describing the costs ---------------------------------
    %            |  Predicted 
    %            |    N    Y     
    %    ------------------------
    %    Actual N | [-1.0  4.0 ;   
    %    Actual Y |   2.0 -2.0  ]   
    %    ------------------------
    %    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %    cost(1,1): Cost of True  Negatives 
    %    cost(2,1): Cost of False Negatives 
    %    cost(1,2): Cost of False Positives
    %    cost(2,2): Cost of True  Positives
    %    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %    Example: cost = [-1, 4; 2 -4]
    %
    %2.  fixed_cost: cost to simply run the algorithm? ---------------------------
    %    Let's assume that one of the inputs to your
    %    algorithm is an Amazon mechanical turker (AMT) worker's rating
    %    of something. Given that AMT workers must be paid, there is a 
    %    fixed overhead cost,  every time you want to run the algorithm.
    %    this fixed cost is what you report here.
    %    Example: fixed_cost = 5.00
    %
    %3. predictions: propobabilities of the true class from the model -----------
    %    Example: predictions = [.1 .2 .3 .2 .6 .9 .9 .9 .8]
    %
    %4. labels: the actual class from the model ---------------------------------
    %    Example: labels =      [ 0  0  0  1  0  0  1  1  1 ]
    %% OUTPUTS(4): %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %   1. X: The probability thresholds (use for plotting curve)
    %
    %   2. Y: The Costs (use for plotting curve)
    %
    %   1. optimal_threshold: The optimal threshold for classification.
    %
    %   2. optimal_cost: The cost corresponding to the optimal threshold.



    %% ALGORITHM : %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % We will explore varous clasification tresholds, and score the confusion 
    % matrix according to our cost matrix.
    ind = 1; step = -1:.01:1;
    for i = step 

      %Adjust the predictions with the new threshold.
      pred_mod = min(max(predictions-i,0),1);

      %compute the confusion matrix
      [cm,order] = confusionmat(1*labels, round(pred_mod));

      %Extract the performance with the new classigfication threshold
      True_Negatives  = cm(1,1); False_Positives = cm(1,2);
      Flase_Negatives = cm(2,1); True_Positives = cm(2,2);

      %Get the fixed costs required to run the mdoel
      %i.e. the costs to simply run the model
      total = sum(sum(cm));
      mdl_fixed_costs = total* fixed_cost; 

      %Get the costs as a fucntion of performance
      mdl_elastic_costs = True_Negatives  * cost(1,1) +...
                          False_Positives * cost(1,2) +...
                          Flase_Negatives * cost(2,1) +...
                          True_Positives  * cost(2,2);

      %total costs
      total_costs(ind) = mdl_elastic_costs + mdl_fixed_costs;

      ind = ind + 1;
    end

    %Identify the optimal threshold and cost.
    [optimal_cost, ind_thresh] = min(total_costs);
    optimal_threshold = 0.5 + step(ind_thresh);

    %Generate a plot showing the cost curve...
    %the cost curve...
    X = 0.5+step;
    Y = total_costs;
    plot(X,Y,'LineWidth',2);
    xlim([0 1]);
    xlabel('Classification Threshold Probability');
    ylabel('Expected Cost');
    hold on;

    %zero cost line...
    plot([0 1],[0 0],'black--','Linewidth',2);
    plot(optimal_threshold, optimal_cost,'o', 'color','k','MarkerFaceColor','k');

    %and the optimal point...
    text(optimal_threshold + 0.01,optimal_cost,'Optimal Threshold');
end