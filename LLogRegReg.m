function errRate = LLogRegReg(designMatrix, stdY,testX)


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
   % ----------------------------------------------------------------------

end
