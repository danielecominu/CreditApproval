function errRate = LDA(designMatrix, stdY)

   % -------- Linear Discriminant Analysis --------------------------------
    global DEBUG
   
   try
        result = fitcdiscr(designMatrix, stdY);
   catch EX
       result = fitcdiscr(designMatrix, stdY, 'discrimType', 'pseudoLinear');
   end
   ee = crossval(result);
   allLDA = kfoldLoss(ee, 'mode', 'individual', 'folds', 5);
   avgLDA = mean(allLDA);
   
   if DEBUG
        disp('-------- Linear Discriminant Analysis --------------------------------');
        fprintf('\n\terror rate:\t%f\n\n',avgLDA);
        fprintf('----------------------------------------------------------------------\n\n');
   end
        
   errRate = avgLDA;
   
   % ----------------------------------------------------------------------

end
