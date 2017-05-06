function errRate =  QLogReg(designMatrix, stdY)

   % -------- Logistic Regression with Quadratic Decision Boundary --------
    global DEBUG
   
  LOGREG = @(XTRAIN, YTRAIN, XTEST, YTEST) logReg(XTRAIN, YTRAIN, XTEST, YTEST, 0);
   
   designMatrix2 = quadMatrix(designMatrix);
   
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



