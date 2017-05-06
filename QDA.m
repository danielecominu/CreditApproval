function errRate = QDA(designMatrix, stdY)

   % -------- Quadratic Discriminant Analysis -----------------------------
    global DEBUG

   % pseudoQuadratic poich? tutte le istanze che per y hanno '-', per la
   % feature g hanno un valore diverso da gg, dunque la varianza di tale
   % insieme ? nulla
   result = fitcdiscr(designMatrix, stdY, 'DiscrimType', 'pseudoQuadratic');
   ee = crossval(result);
   allQDA = kfoldLoss(ee, 'mode', 'individual', 'folds', 5);
   avgQDA = mean(allQDA);

   if DEBUG
        disp('-------- Quadratic Discriminant Analysis -----------------------------');
        fprintf('\n\terror rate:\t%f\n\n',avgQDA);
        fprintf('----------------------------------------------------------------------\n\n');
   end

   errRate = avgQDA;

   % ----------------------------------------------------------------------
end