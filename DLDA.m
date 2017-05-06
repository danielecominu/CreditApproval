function predictions = DLDA(designMatrix, stdY, testX)
    
   % -------- Linear Discriminant Analysis --------------------------------
    global DEBUG
   
   model = fitcdiscr(designMatrix, stdY, 'DiscrimType', 'diagLinear');
   
   predictions = predict(model, testX);
   % ----------------------------------------------------------------------
end