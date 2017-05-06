function errRate = DLDA(designMatrix, stdY)
    
   % -------- Linear Discriminant Analysis --------------------------------
    global DEBUG
   
   result = fitcdiscr(designMatrix, stdY, 'DiscrimType', 'diagLinear');
   ee = crossval(result);
   allDLDA = kfoldLoss(ee, 'mode', 'individual', 'folds', 5);
   avgDLDA = mean(allDLDA);
   
   if DEBUG
        disp('-------- Diagonal Linear Discriminant Analysis --------------------------------');
        fprintf('\n\terror rate:\t%f\n\n',avgDLDA);
        fprintf('----------------------------------------------------------------------\n\n');
   end
   
   errRate = avgDLDA;
   
   % ----------------------------------------------------------------------
end