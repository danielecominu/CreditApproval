function Validator()
    
   M = load('dati.mat'); 
   Numeric = [M.VarName11 M.VarName14 M.VarName15 M.VarName2 M.VarName3 M.VarName8];
   stdNumeric = standardize(Numeric);
   oh_b = oneHotEncoding(M.b);
   oh_f = oneHotEncoding(M.f);
   oh_g = oneHotEncoding(M.g);
   oh_g1 = oneHotEncoding(M.g1);
   oh_t = oneHotEncoding(M.t);
   oh_t1 = oneHotEncoding(M.t1);
   oh_u = oneHotEncoding(M.u);
   oh_v = oneHotEncoding(M.v);
   oh_w = oneHotEncoding(M.w);
   
   designMatrix = [stdNumeric, oh_b, oh_f, oh_g, oh_g1, oh_t, oh_t1, oh_u, oh_v, oh_w];
   stdY = editY(M.VarName16);
   
   % pseudoQuadratic poich? tutte le istanze che per y hanno '-', per la
   % feature g hanno un valore diverso da gg, dunque la varianza di tale
   % insieme ? nulla
   result = fitcdiscr(designMatrix, stdY, 'DiscrimType', 'pseudoQuadratic');
   ee = crossval(result);
   allQDA = kfoldLoss(ee, 'mode', 'individual');
   avgQDA = mean(allQDA)
   
   result = fitcdiscr(designMatrix, stdY);
   ee = crossval(result);
   allLDA = kfoldLoss(ee, 'mode', 'individual');
   avgLDA = mean(allLDA)
   
   % trainingSize = 550
   
   % trainingX = designMatrix(1:trainingSize, :);
   % trainingY = stdY(1:trainingSize);
   % testX = designMatrix(trainingSize:end, :);
   % testY = stdY(trainingSize:end);
   
   % model = logregFit(trainingX, trainingY);
   % [yhat, prob] = logregPredict(model, testX);
   % errRateLogReg = 1 - (sum(yhat == testY)/rows(testY))
   
   % using 10-fold cross validation
   
  LOGREG = @(XTRAIN, YTRAIN, XTEST, YTEST) logReg(XTRAIN, YTRAIN, XTEST, YTEST);
   

   allLogReg = crossval(LOGREG, designMatrix, stdY);
   avgLogReg = mean(allLogReg)
   
   ls = linspace(0,20,500);
   best = 0;
   avgLogRegR = 1;
   for i=1:500
       cvL = cvLambda(designMatrix, stdY, ls(i));
       if(cvL < avgLogRegR)
           best = ls(i);
           avgLogRegR = cvL;
       end
   end
   avgLogRegR

end

function errRate = logReg(trainingX, trainingY, testX, testY)
    
   model = logregFit(trainingX, trainingY);
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

function vEnc = oneHotEncoding(V)

    X = unique(V);
    [~, loc] = ismember(V, X); %per ogni elemento, mi prendo l'indice.
    vEnc = double(loc == 1:size(X));
end
