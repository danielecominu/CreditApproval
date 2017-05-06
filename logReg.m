function errRate = logReg(trainingX, trainingY, testX, testY, lambda)
    
   model = logregFit(trainingX, trainingY, 'regType', 'L2', 'lambda', lambda);
   [yhat, prob] = logregPredict(model, testX);
   errRate = 1 - (sum(yhat == testY)/rows(testY));
   
end