function predictions = QLogRegReg(designMatrix, stdY, testX)
   
   % Logistic Regression with Quadratic Decision Boundary and Regularization
   
   designMatrix2 = quadMatrix(designMatrix);
   
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
   
    model = logregFit(designMatrix2, stdY, 'lambda', best,'regType','L2');
   [predictions, ~] = logregPredict(model, quadMatrix(testX));
   % ----------------------------------------------------------------------

end