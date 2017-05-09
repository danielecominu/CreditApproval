function predictions =  QLogReg(designMatrix, stdY, testX)

   % -------- Logistic Regression with Quadratic Decision Boundary --------
    
   designMatrix2 = quadMatrix(designMatrix);
   model = logregFit(designMatrix2, stdY);
   [predictions, ~] = logregPredict(model, quadMatrix(testX));
   
   % ----------------------------------------------------------------------

end



