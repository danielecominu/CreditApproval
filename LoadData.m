function [designMatrix, stdY] = LoadData()
    
   M = load('dati.mat'); 
   Numeric = [M.VarName11 M.VarName14 M.VarName15 M.VarName2 M.VarName3 M.VarName8];
   oh_b = oneHotEncoding(M.b);
   oh_f = oneHotEncoding(M.f);
   oh_g = oneHotEncoding(M.g);
   oh_g1 = oneHotEncoding(M.g1);
   oh_t = oneHotEncoding(M.t);
   oh_t1 = oneHotEncoding(M.t1);
   oh_u = oneHotEncoding(M.u);
   oh_v = oneHotEncoding(M.v);
   oh_w = oneHotEncoding(M.w);
   
   designMatrix = [Numeric, oh_b, oh_f, oh_g, oh_g1, oh_t, oh_t1, oh_u, oh_v, oh_w];
   designMatrix = standardize(designMatrix);
   stdY = editY(M.VarName16);

end
    
function [stdMatrix] = standardize(X)
    
    mu = mean(X);              % returns a row vector
    sigma = std(X);            % returns a row vector
    m = size(X, 1);            % returns the number of rows in X
    mu_matrix = ones(m, 1) * mu;  
    sigma_matrix = ones(m, 1) * sigma;
    stdMatrix = X;
    stdMatrix = bsxfun(@minus, stdMatrix, mu_matrix);
    stdMatrix = bsxfun(@rdivide, stdMatrix, sigma_matrix);
end

function [new_y] = editY(Y)
    
    new_y = [];
    Y = cell2mat(Y);
    new_y = Y == '+';
end

function vEnc = toNumber(V) 
    X = unique(V);
    [~, vEnc] = ismember(V, X); %per ogni elemento, mi prendo l'indice.
end

function vEnc = oneHotEncoding(V)

    X = unique(V);
    [~, loc] = ismember(V, X); %per ogni elemento, mi prendo l'indice.
    vEnc = double(bsxfun(@eq, loc, 1:size(X)));
end
