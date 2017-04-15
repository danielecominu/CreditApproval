function Validator()

    global DEBUG
    DEBUG = 1;
    
    if DEBUG == 0
        warning('off','last');
    end
    
   close all;
    
   [X, Y] = loadData();
   
   START_PCA = 35;
   END_PCA = 35;
   
   EXECUTIONS = 10;
   
   POINT_SIZE = 10;
   
   % Principal Component Analysis
   [coeff, X, pcVariance] = pca(X);
   
   pcVariance = pcVariance ./ (pcVariance(1, 1)*5);
   
  % plot(1:size(pcVariance), pcVariance)
   hold on
   
   errRates = ones(16, 7, 5);
   
   LFM = 0;
   EFM = 1;
   
    [tvX, stdY, testX, testY] = divide(X, Y); 
    model = logregFit(tvX, stdY, 'regType', 'L2', 'lambda', 20);
    [yhat, prob] = logregPredict(model, testX);
    err = 1 - (sum(yhat == testY)/rows(testY));
    
    fprintf("Final error after training the whole dataset is %f\n", err)
   %return
   % itero diverse volte gli algoritmi con dataset diversi
   for j=1:EXECUTIONS
       
       % per ottenere sempre dati diversi applico ogni volta una
       % permutazione casuale e prendo solo i primi nonTestSamples samples
       [tvX, stdY, ~, ~] = divide(X, Y); 
       
       for c=START_PCA:END_PCA
   %scatter( [20     24      25      26      27      28      29      30      31      32      33      34      35      46], ...
   %         [0.1270 0.1224  0.1179  0.1179  0.1163  0.1149  0.1195  0.1193  0.1178  0.1178  0.1162  0.1194  0.1194  0.1210])
   %set(gca,'XTick',1:size(pcVariance));
   
   %hold off
   %figure
   
   %gscatter(newDesignMatrix(:, 1), newDesignMatrix(:, 2), stdY, 'rb','..', 7, 'off')
   
            fprintf('run n?: %d, n? features: %d\n', j, c)
   
            designMatrix = tvX(:, 1:c);
    
            errRates(c-START_PCA+1, 1, j) = DLDA(designMatrix, stdY);
            errRates(c-START_PCA+1, 2, j) = LDA(designMatrix, stdY);
            errRates(c-START_PCA+1, 3, j) = QDA(designMatrix, stdY);
            errRates(c-START_PCA+1, 4, j) = LLogReg(designMatrix, stdY);
            errRates(c-START_PCA+1, 5, j) = QLogReg(designMatrix, stdY);
            [errRates(c-START_PCA+1, 6, j), cLambda] = LLogRegReg(designMatrix, stdY);
            errRates(c-START_PCA+1, 7, j) = QLogRegReg(designMatrix, stdY);
            
            if(errRates(c-START_PCA+1, 6, j) < EFM)
                EFM = errRates(c-START_PCA+1, 6, j);
                LFM = cLambda;
            end
            
            gscatter(ones(7, 1).*c, errRates(c-START_PCA+1, :, j), 1:7, 'ymcrgbk','.......', POINT_SIZE, 'off')
        
       end
   end
   
   VARIANCE_POINT_SIZE = 2400;
   
   for c=START_PCA:END_PCA
       err_rates = squeeze(errRates(c-START_PCA+1, :, :))';
       mean_ = mean(err_rates);
       std_ = std(err_rates);
       
       gscatter(c, mean_(1), 1, 'y', 'o', VARIANCE_POINT_SIZE*std_(1), 'off');
       gscatter(c, mean_(2), 2, 'm', 'o', VARIANCE_POINT_SIZE*std_(2), 'off');
       gscatter(c, mean_(3), 3, 'c', 'o', VARIANCE_POINT_SIZE*std_(3), 'off');
       gscatter(c, mean_(4), 4, 'r', 'o', VARIANCE_POINT_SIZE*std_(4), 'off');
       gscatter(c, mean_(5), 5, 'g', 'o', VARIANCE_POINT_SIZE*std_(5), 'off');
       gscatter(c, mean_(6), 6, 'b', 'o', VARIANCE_POINT_SIZE*std_(6), 'off');
       gscatter(c, mean_(7), 7, 'k', 'o', VARIANCE_POINT_SIZE*std_(7), 'off');
   % gscatter(START_PCA:END_PCA, mean(err_rates)')
   end
   
   set(gca,'XTick',1:size(pcVariance));
   legend('Diagonal Linear Discriminant Analysis', 'Linear Discriminant Analysis', ...
            'Quadratic Discriminant Analysis', 'Logistic Regression (Linear Boundary)', 'Logistic Regression (Quadratic Boundary)', ...
            'Logistic Regression (Linear Boundary) with Regularization', 'Logistic Regression (Quadratic Boundary) with Regularization');
   
   figure
   
   
   
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
   
   
    model = logregFit(trainingX, trainingY, 'regType', 'L2', 'lambda', LFM);
    [yhat, prob] = logregPredict(model, testX);
    err(k) = 1 - (sum(yhat == testY)/rows(testY));
end

function [designMatrix, stdY, testX, testY] = divide(X, Y)

   samples = size(X, 1);

    perm = randperm(samples);
    X = X(perm, :);
    Y = Y(perm, :);
    
    nonTestSamples = int64(samples*0.8);
    testSamples = samples - nonTestSamples;
        
    testX = X(nonTestSamples:end, :);
    designMatrix = X(1:nonTestSamples, :);
    testY = Y(nonTestSamples:end, :);
    stdY = Y(1:nonTestSamples, :);

end

function [designMatrix, stdY] = loadData()
    
   M = load('dati.mat'); 
   Numeric = [M.VarName11 M.VarName14 M.VarName15 M.VarName2 M.VarName3 M.VarName8];
   oh_b = oneHotEncoding(M.b);
   oh_f = oneHotEncoding(M.f);
   oh_g = oneHotEncoding(M.g);
   oh_g1 = oneHotEncoding(M.g1);
   oh_t = oneHotEncoding(M.t);
   oh_t1 = oneHotEncoding(M.t1);
   oh_u = oneHotEncoding(M.u);
   oh_v = oneHotEncoding(M.v);
   oh_w = oneHotEncoding(M.w);
   
   designMatrix = [Numeric, oh_b, oh_f, oh_g, oh_g1, oh_t, oh_t1, oh_u, oh_v, oh_w];
   designMatrix = standardize(designMatrix);
   stdY = editY(M.VarName16);

end


function errRate = DLDA(designMatrix, stdY)
    
   % -------- Linear Discriminant Analysis --------------------------------
    global DEBUG
   
   result = fitcdiscr(designMatrix, stdY, 'DiscrimType', 'diagLinear');
   ee = crossval(result);
   allDLDA = kfoldLoss(ee, 'mode', 'individual', 'folds', 5);
   avgDLDA = mean(allDLDA);
   
   if DEBUG
        disp('-------- Diagonal Linear Discriminant Analysis --------------------------------');
        fprintf('\n\terror rate:\t%f\n\n',avgDLDA);
        fprintf('----------------------------------------------------------------------\n\n');
   end
   
   errRate = avgDLDA;
   
   % ----------------------------------------------------------------------
end


function errRate = LDA(designMatrix, stdY)

   % -------- Linear Discriminant Analysis --------------------------------
    global DEBUG
   
   try
        result = fitcdiscr(designMatrix, stdY);
   catch EX
       result = fitcdiscr(designMatrix, stdY, 'discrimType', 'pseudoLinear');
   end
   ee = crossval(result);
   allLDA = kfoldLoss(ee, 'mode', 'individual', 'folds', 5);
   avgLDA = mean(allLDA);
   
   if DEBUG
        disp('-------- Linear Discriminant Analysis --------------------------------');
        fprintf('\n\terror rate:\t%f\n\n',avgLDA);
        fprintf('----------------------------------------------------------------------\n\n');
   end
        
   errRate = avgLDA;
   
   % ----------------------------------------------------------------------

end

function errRate = QDA(designMatrix, stdY) 

   % -------- Quadratic Discriminant Analysis -----------------------------
    global DEBUG
   
   % pseudoQuadratic poich? tutte le istanze che per y hanno '-', per la
   % feature g hanno un valore diverso da gg, dunque la varianza di tale
   % insieme ? nulla
   result = fitcdiscr(designMatrix, stdY, 'DiscrimType', 'pseudoQuadratic');
   ee = crossval(result);
   allQDA = kfoldLoss(ee, 'mode', 'individual', 'folds', 5);
   avgQDA = mean(allQDA);
   
   if DEBUG
        disp('-------- Quadratic Discriminant Analysis -----------------------------');
        fprintf('\n\terror rate:\t%f\n\n',avgQDA);
        fprintf('----------------------------------------------------------------------\n\n');
   end
        
   errRate = avgQDA;
   
   % ----------------------------------------------------------------------
end

function errRate = LLogReg(designMatrix, stdY)
    % -------- Logistic Regression with Linear Decision Boundary -----------
    global DEBUG
   
  LOGREG = @(XTRAIN, YTRAIN, XTEST, YTEST) logReg(XTRAIN, YTRAIN, XTEST, YTEST, 0);
   
   allLogReg = crossval(LOGREG, designMatrix, stdY, 'kfold', 5);
   avgLogReg = mean(allLogReg);
   
   if DEBUG
        disp('-------- Logistic Regression with Linear Decision Boundary -----------');
        fprintf('\n\terror rate:\t%f\n\n',avgLogReg);
        fprintf('----------------------------------------------------------------------\n\n');
   end
   
   errRate = avgLogReg;
   
   % ----------------------------------------------------------------------
end

function quadMatrix = quad(designMatrix)
    designMatrix2 = designMatrix;
    
   [~, features] = size(designMatrix);
   
    for i=1:features
       col = designMatrix(:, i);
       matr = col(:, ones(1, features-i+1));
       quadMatrix = designMatrix(:, i:features).*matr;
       designMatrix2 = [designMatrix2, quadMatrix];
    end
    
    quadMatrix = designMatrix2;
end

function errRate =  QLogReg(designMatrix, stdY)

   % -------- Logistic Regression with Quadratic Decision Boundary --------
    global DEBUG
   
  LOGREG = @(XTRAIN, YTRAIN, XTEST, YTEST) logReg(XTRAIN, YTRAIN, XTEST, YTEST, 0);
   
   designMatrix2 = quad(designMatrix);
   
   allLogReg2 = crossval(LOGREG, designMatrix2, stdY, 'kfold', 5);
   avgLogReg2 = mean(allLogReg2);
   
   if DEBUG
        disp('-------- Logistic Regression with Quadratic Decision Boundary --------');
        fprintf('\n\terror rate:\t%f\n\n',avgLogReg2);
        fprintf('----------------------------------------------------------------------\n\n');
   end
   
   errRate = avgLogReg2;
   
   % ----------------------------------------------------------------------

end

function [errRate, Blambda] = LLogRegReg(designMatrix, stdY)


   % Logistic Regression with Linear Decision Boundary and Regularization -
   
   ls = linspace(0,20,500);
   best = 0;
   avgLogRegR = 1;
   for i=1:500
       
       logRegLambda = @(XTRAIN, YTRAIN, XTEST, YTEST) logReg(XTRAIN, YTRAIN, XTEST, YTEST, ls(i));
   
       allLogRegR = crossval(logRegLambda, designMatrix, stdY, 'kfold', 5);
       cvL = mean(allLogRegR);
       
       %cvL = cvLambda(designMatrix, stdY, ls(i));
       if(cvL < avgLogRegR)
           best = ls(i);
           avgLogRegR = cvL;
       end
   end
   
   errRate = avgLogRegR;
   Blambda = best;
   % ----------------------------------------------------------------------

end

function errRate = QLogRegReg(designMatrix, stdY)
   
   % Logistic Regression with Quadratic Decision Boundary and Regularization
   
   designMatrix2 = quad(designMatrix);
   
   ls = linspace(15,17,10);
   best = 0;
   avgLogRegR = 1;
   for i=1:10
       
       logRegLambda = @(XTRAIN, YTRAIN, XTEST, YTEST) logReg(XTRAIN, YTRAIN, XTEST, YTEST, ls(i));
   
       allLogRegR = crossval(logRegLambda, designMatrix2, stdY, 'kfold', 5);
       cvL = mean(allLogRegR);
       
       %cvL = cvLambda(designMatrix, stdY, ls(i));
       if(cvL < avgLogRegR)
           best = ls(i);
           avgLogRegR = cvL;
       end
   end
   
   errRate = avgLogRegR;
   
   % ----------------------------------------------------------------------

end

function errRate = logReg(trainingX, trainingY, testX, testY, lambda)
    
   model = logregFit(trainingX, trainingY, 'regType', 'L2', 'lambda', lambda);
   [yhat, prob] = logregPredict(model, testX);
   errRate = 1 - (sum(yhat == testY)/rows(testY));
   
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

function [stdMatrix] = standardize(X)
    
    mu = mean(X);              % returns a row vector
    sigma = std(X);            % returns a row vector
    m = size(X, 1);            % returns the number of rows in X
    mu_matrix = ones(m, 1) * mu;  
    sigma_matrix = ones(m, 1) * sigma;
    stdMatrix = X;
    stdMatrix = bsxfun(@minus, stdMatrix, mu_matrix);
    stdMatrix = bsxfun(@rdivide, stdMatrix, sigma_matrix);
end

function [new_y] = editY(Y)
    
    new_y = [];
    Y = cell2mat(Y);
    new_y = Y == '+';
end

function vEnc = toNumber(V) 
    X = unique(V);
    [~, vEnc] = ismember(V, X); %per ogni elemento, mi prendo l'indice.
end

function vEnc = oneHotEncoding(V)

    X = unique(V);
    [~, loc] = ismember(V, X); %per ogni elemento, mi prendo l'indice.
    vEnc = double(loc == 1:size(X));
end
