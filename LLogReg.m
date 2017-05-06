function errRate = LLogReg(designMatrix, stdY,testX)
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
