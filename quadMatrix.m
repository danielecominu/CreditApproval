function quadMatrix = quadMatrix(designMatrix)
    designMatrix2 = designMatrix;
    
   [~, features] = size(designMatrix);
   
    for i=1:features
       col = designMatrix(:, i);
       matr = col(:, ones(1, features-i+1));
       quadMatrix = designMatrix(:, i:features).*matr;
       designMatrix2 = [designMatrix2, quadMatrix];
    end
    
    quadMatrix = designMatrix2;
end