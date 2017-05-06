function predictions = LLogReg(designMatrix, stdY, testX)
    % -------- Logistic Regression with Linear Decision Boundary -----------   
    model = logregFit(designMatrix, stdY);
    [predictions, ~] = logregPredict(model, testX);
    
   % ----------------------------------------------------------------------
end