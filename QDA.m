function predictions = LDA(designMatrix, stdY, testX)

   % -------- Linear Discriminant Analysis --------------------------------
    global DEBUG
   
   try
        model = fitcdiscr(designMatrix, stdY);
   catch EX
       model = fitcdiscr(designMatrix, stdY, 'discrimType', 'pseudoLinear');
   end
   
   predictions = predict(model, testX);
   
   
   % ----------------------------------------------------------------------

end
