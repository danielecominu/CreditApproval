function predictions = QDA(designMatrix, stdY, testX)

   % -------- Quadratic Discriminant Analysis -----------------------------
   
   try
        model = fitcdiscr(designMatrix, stdY);
   catch EX
       model = fitcdiscr(designMatrix, stdY, 'discrimType', 'pseudoLinear');
   end

   predictions = predict(model, testX);

   
   % ----------------------------------------------------------------------

end
