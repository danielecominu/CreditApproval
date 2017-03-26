function Validator()
    
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
   
   % Principal Component Analysis
   [coeff, newDesignMatrix, pcVariance] = pca(designMatrix);
   
   pcVariance = pcVariance ./ (pcVariance(1, 1)*5);
   
   plot(1:size(pcVariance), pcVariance)
   hold on
   scatter( [20     24      25      26      27      28      29      30      31      32      33      34      35      46], ...
            [0.1270 0.1224  0.1179  0.1179  0.1163  0.1149  0.1195  0.1193  0.1178  0.1178  0.1162  0.1194  0.1194  0.1210])
   set(gca,'XTick',1:size(pcVariance));
   
   hold off
   figure
   
   gscatter(newDesignMatrix(:, 1), newDesignMatrix(:, 2), stdY, 'rb','..', 7, 'off')
   
   % 20 -> 0.1270
   % 24 -> 0.1224
   % 25 -> 0.1179
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
   % without pca -> 0.1210
   designMatrix = newDesignMatrix(:, 1:28);
   
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
   
   %return
   
   %{
   for i=1:6
        scatter(stdNumeric(:, i), stdY)
        title(sprintf('Feature %d', i))
        figure
   end
   %}
   
   % -------- Linear Discriminant Analysis --------------------------------
   
   result = fitcdiscr(designMatrix, stdY, 'DiscrimType', 'diagLinear');
   ee = crossval(result);
   allDLDA = kfoldLoss(ee, 'mode', 'individual');
   avgDLDA = mean(allDLDA);
   
   disp('-------- Diagonal Linear Discriminant Analysis --------------------------------');
   fprintf('\n\terror rate:\t%f\n\n',avgDLDA);
   fprintf('----------------------------------------------------------------------\n\n');
   
   % ----------------------------------------------------------------------
   
   
   % -------- Linear Discriminant Analysis --------------------------------
   
   result = fitcdiscr(designMatrix, stdY);
   ee = crossval(result);
   allLDA = kfoldLoss(ee, 'mode', 'individual');
   avgLDA = mean(allLDA);
   
   disp('-------- Linear Discriminant Analysis --------------------------------');
   fprintf('\n\terror rate:\t%f\n\n',avgLDA);
   fprintf('----------------------------------------------------------------------\n\n');
   
   % ----------------------------------------------------------------------
   
   
   % -------- Quadratic Discriminant Analysis -----------------------------
   
   % pseudoQuadratic poich? tutte le istanze che per y hanno '-', per la
   % feature g hanno un valore diverso da gg, dunque la varianza di tale
   % insieme ? nulla
   result = fitcdiscr(designMatrix, stdY, 'DiscrimType', 'pseudoQuadratic');
   ee = crossval(result);
   allQDA = kfoldLoss(ee, 'mode', 'individual');
   avgQDA = mean(allQDA);
   
   disp('-------- Quadratic Discriminant Analysis -----------------------------');
   fprintf('\n\terror rate:\t%f\n\n',avgQDA);
   fprintf('----------------------------------------------------------------------\n\n');
   
   % ----------------------------------------------------------------------
   
   
   % trainingSize = 550
   
   % trainingX = designMatrix(1:trainingSize, :);
   % trainingY = stdY(1:trainingSize);
   % testX = designMatrix(trainingSize:end, :);
   % testY = stdY(trainingSize:end);
   
   % model = logregFit(trainingX, trainingY);
   % [yhat, prob] = logregPredict(model, testX);
   % errRateLogReg = 1 - (sum(yhat == testY)/rows(testY))
   
   % using 10-fold cross validation
   %}
   
  LOGREG = @(XTRAIN, YTRAIN, XTEST, YTEST) logReg(XTRAIN, YTRAIN, XTEST, YTEST, 0);
   

   % -------- Logistic Regression with Linear Decision Boundary -----------
   
   allLogReg = crossval(LOGREG, designMatrix, stdY);
   avgLogReg = mean(allLogReg);
   
   disp('-------- Logistic Regression with Linear Decision Boundary -----------');
   fprintf('\n\terror rate:\t%f\n\n',avgLogReg);
   fprintf('----------------------------------------------------------------------\n\n');
   % ----------------------------------------------------------------------
   
   % -------- Logistic Regression with Quadratic Decision Boundary --------
   
   designMatrix2 = designMatrix;
   for i=1:features
       col = designMatrix(:, i);
       matr = col(:, ones(1, features-i+1));
       quadMatrix = designMatrix(:, i:features).*matr;
       designMatrix2 = [designMatrix2, quadMatrix];
   end
   
   allLogReg2 = crossval(LOGREG, designMatrix2, stdY);
   avgLogReg2 = mean(allLogReg2);
   
   disp('-------- Logistic Regression with Quadratic Decision Boundary --------');
   fprintf('\n\terror rate:\t%f\n\n',avgLogReg2);
   fprintf('----------------------------------------------------------------------\n\n');
   
   % ----------------------------------------------------------------------
   
   
   
   % Logistic Regression with Linear Decision Boundary and Regularization -
   
   ls = linspace(0,20,500);
   best = 0;
   avgLogRegR = 1;
   for i=1:500
       
       logRegLambda = @(XTRAIN, YTRAIN, XTEST, YTEST) logReg(XTRAIN, YTRAIN, XTEST, YTEST, ls(i));
   
       allLogRegR = crossval(logRegLambda, designMatrix, stdY);
       cvL = mean(allLogRegR);
       
       %cvL = cvLambda(designMatrix, stdY, ls(i));
       if(cvL < avgLogRegR)
           best = ls(i);
           avgLogRegR = cvL;
       end
   end
   avgLogRegR
   % ----------------------------------------------------------------------
   
   
   % Logistic Regression with Quadratic Decision Boundary and Regularization
   
   ls = linspace(15,17,10);
   best = 0;
   avgLogRegR = 1;
   for i=1:10
       
       logRegLambda = @(XTRAIN, YTRAIN, XTEST, YTEST) logReg(XTRAIN, YTRAIN, XTEST, YTEST, ls(i));
   
       allLogRegR = crossval(logRegLambda, designMatrix2, stdY);
       cvL = mean(allLogRegR);
       
       %cvL = cvLambda(designMatrix, stdY, ls(i));
       if(cvL < avgLogRegR)
           best = ls(i);
           avgLogRegR = cvL;
       end
   end
   avgLogRegR
   % ----------------------------------------------------------------------
   
    
   %{
   trainingSize = 550;
   
   trainingX = designMatrix2(1:trainingSize, :);
   trainingY = stdY(1:trainingSize);
   testX = designMatrix2(trainingSize:end, :);
   testY = stdY(trainingSize:end);
   
   logReg(trainingX, trainingY, testX, testY, 15.22);   
   %}
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
