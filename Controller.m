function Controller()

    global DEBUG
    DEBUG = 1;
    
    close all;
    
   [X, Y] = LoadData();
   
   START_PCA = 22;
   END_PCA = 32;
   
   EXECUTIONS = 10;
  
   MODELS = 7;
   
   POINT_SIZE = 10;
   
   % Principal Component Analysis
   [coeff, X, pcVariance] = pca(X);
   
   pcVariance = pcVariance ./ (pcVariance(1, 1)*5);
   
   plot(1:size(pcVariance), pcVariance)
   hold on
   
   errRates = ones(END_PCA - START_PCA + 1, MODELS, EXECUTIONS);
   
   [X,Y, testX, testY] = divide(X, Y, .8);
   
   TEST_SIZE = size(testX,1);
   biasMatrix = zeros(TEST_SIZE, EXECUTIONS, MODELS);
   
   % itero diverse volte gli algoritmi con dataset diversi
   for j=1:EXECUTIONS
       
       % per ottenere sempre dati diversi applico ogni volta una
       % permutazione casuale e prendo solo i primi nonTestSamples samples
       [tvX, stdY, ~, ~] = divide(X, Y, .9);
        
       for c=START_PCA:END_PCA
   %scatter( [20     24      25      26      27      28      29      30      31      32      33      34      35      46], ...
   %         [0.1270 0.1224  0.1179  0.1179  0.1163  0.1149  0.1195  0.1193  0.1178  0.1178  0.1162  0.1194  0.1194  0.1210])
   %set(gca,'XTick',1:size(pcVariance));
   
   %hold off
   %figure
   
   %gscatter(newDesignMatrix(:, 1), newDesignMatrix(:, 2), stdY, 'rb','..', 7, 'off')
   
            fprintf('run n?: %d, n? features: %d\n', j, c)
   
            designMatrix = tvX(:, 1:c);
            pcaTestX = testX(:, 1:c);
            
            ErrRates = @(xPred) sum(abs(xPred - testY)) / TEST_SIZE;
    
            %errRates(c-START_PCA+1, 1, j) = DLDA(designMatrix, stdY, pcaTestX);
            y = DLDA(designMatrix, stdY, pcaTestX);
            biasMatrix(:, j, 1) = biasMatrix(:, j, 1) + y;
            errRates(c-START_PCA+1, 1, j) = ErrRates(y);
            
            y = LDA(designMatrix, stdY, pcaTestX);
            biasMatrix(:, j, 2) = biasMatrix(:, j, 2) + y;
            errRates(c-START_PCA+1, 2, j) = ErrRates(y);
            
            y = QDA(designMatrix, stdY, pcaTestX);
            biasMatrix(:, j, 3) = biasMatrix(:, j, 3) + y; 
            errRates(c-START_PCA+1, 3, j) = ErrRates(y);
            
            y = LLogReg(designMatrix, stdY, pcaTestX);
            biasMatrix(:, j, 4) = biasMatrix(:, j, 4) + y;
            errRates(c-START_PCA+1, 4, j) = ErrRates(y);
            
            y = QLogReg(designMatrix, stdY, pcaTestX);
            biasMatrix(:, j, 5) = biasMatrix(:, j, 5) + y;
            errRates(c-START_PCA+1, 5, j) = ErrRates(y);
            
            y = LLogRegReg(designMatrix, stdY, pcaTestX);
            biasMatrix(:, j, 6) = biasMatrix(:, j, 6) + y;
            errRates(c-START_PCA+1, 6, j) = ErrRates(y);
            
            y = QLogRegReg(designMatrix, stdY, pcaTestX);
            biasMatrix(:, j, 7) = biasMatrix(:, j, 7) + y;
            errRates(c-START_PCA+1, 7, j) = ErrRates(y);
            
            gscatter(ones(MODELS).*c, errRates(c-START_PCA+1, :, j), 1:MODELS, 'gmcrybk','.......', POINT_SIZE, 'off')
        
       end
   end
    
   biasMatrix(:,:,:) = biasMatrix(:,:,:) ./ (END_PCA - START_PCA +1); 
   bias = zeros(TEST_SIZE, MODELS);
   variance = zeros(TEST_SIZE, MODELS);
   
   for m=1:MODELS
       meanP = mean(biasMatrix(:,:,m), 2);
       bias(:, m) = meanP - testY; %media sulle righe !
       variance(:,m) = mean((biasMatrix(:,:,m) - meanP).^2,2);
       
   end
   
   VARIANCE_POINT_SIZE = 5500;
   
   biasF = mean(bias)
   varianceF = mean(variance)
   figure
   hold on
   gscatter((1:7), biasF, 1:7, 'gmcrybk','.......', POINT_SIZE, 'off')
   gscatter((1:7), biasF, 1:7, 'gmcrybk','ooooooo', VARIANCE_POINT_SIZE*max(varianceF, 0.0001), 'off')
   legend('Diagonal Linear Discriminant Analysis', 'Linear Discriminant Analysis', ...
            'Quadratic Discriminant Analysis', 'Logistic Regression (Linear Boundary)', 'Logistic Regression (Quadratic Boundary)', ...
            'Logistic Regression (Linear Boundary) with Regularization', 'Logistic Regression (Quadratic Boundary) with Regularization');
   
   figure
   
   if 0
   for c=START_PCA:END_PCA
       err_rates = squeeze(errRates(c-START_PCA+1, :, :))';
       mean_ = mean(err_rates);
       std_ = std(err_rates);
       
       gscatter(c, mean_(1), 1, 'g', 'o', VARIANCE_POINT_SIZE*std_(1), 'off');
       gscatter(c, mean_(2), 2, 'm', 'o', VARIANCE_POINT_SIZE*std_(2), 'off');
       gscatter(c, mean_(3), 3, 'c', 'o', VARIANCE_POINT_SIZE*std_(3), 'off');
       gscatter(c, mean_(4), 4, 'r', 'o', VARIANCE_POINT_SIZE*std_(4), 'off');
       gscatter(c, mean_(5), 5, 'y', 'o', VARIANCE_POINT_SIZE*std_(5), 'off');
       gscatter(c, mean_(6), 6, 'b', 'o', VARIANCE_POINT_SIZE*std_(6), 'off');
       gscatter(c, mean_(7), 7, 'k', 'o', VARIANCE_POINT_SIZE*std_(7), 'off');
   % gscatter(START_PCA:END_PCA, mean(err_rates)')
   end
   end
   
   set(gca,'XTick',1:size(pcVariance));
   
   figure
   
   % 20 -> 0.1270
   % 24 -> 0.1224
   % 25 -> 0.1179 0.1129
   % 26 -> 0.1179
   % 27 -> 0.1163
   % 28 -> 0.1149
   % 29 -> 0.1195
   % 30 -> 0.1193
   % 31 -> 0.1178
   % 32 -> 0.1178
   % 33 -> 0.1162
   % 34 -> 0.1194
   % 35 -> 0.1194
   % without pca -> 0.1210 0.1072
   % designMatrix = newDesignMatrix(:, 1:25);
   
   
   
   [~, features] = size(designMatrix);
   
   numFeatures = 5;
   padding = 0.5;
   margin = 0.005;
   pointSize = 3;
   f = 0;
   
   for i=1:numFeatures
        for j=1:numFeatures
            subplot('Position', [(i-1)/numFeatures+margin 1-(j)/numFeatures+margin 1/numFeatures-2*margin 1/numFeatures-2*margin])
            if(i == j)
                gscatter([-900 800]', [800 -400], [0 1], 'gb', '..', pointSize, 'off')
            else
                gscatter(designMatrix(:,i+f), designMatrix(:,j+f), stdY, 'rb','..', pointSize, 'off')
            end
            axis([min(designMatrix(:, i+f))-padding max(designMatrix(:,i+f))+padding min(designMatrix(:, j+f))-padding max(designMatrix(:,j+f))+padding])
            set(gca,'YTick',[]);
            set(gca,'XTick',[]);
            
            %title(sprintf('%d -- %d', i,j))
        end
   end
end

function [designMatrix, stdY, testX, testY] = divide(X, Y, p)

   samples = size(X, 1);

    perm = randperm(samples);
    X = X(perm, :);
    Y = Y(perm, :);
    
    nonTestSamples = int64(samples*p);
    testSamples = samples - nonTestSamples;
        
    testX = X(nonTestSamples:end, :);
    designMatrix = X(1:nonTestSamples, :);
    testY = Y(nonTestSamples:end, :);
    stdY = Y(1:nonTestSamples, :);

end

function errRate = cvLambda(X,Y,lambda) 
    % partition data into 10 folds
    K = 10;
    cv = cvpartition(numel(Y), 'kfold',K);

    err = zeros(K,1);
    for k=1:K
        % training/testing indices for this fold
        trainIdx = cv.training(k);
        testIdx = cv.test(k);

        trainingX = X(trainIdx,:);
        trainingY = Y(trainIdx,:);
        testX = X(testIdx,:);
        testY = Y(testIdx,:);
        
        model = logregFit(trainingX, trainingY, 'regType', 'L2', 'lambda', lambda);
        [yhat, prob] = logregPredict(model, testX);
        err(k) = 1 - (sum(yhat == testY)/rows(testY));
    end

    % average errRate across k-folds
    errRate = mean((err));
end
