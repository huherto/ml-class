function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

h = sigmoid(X * theta);
J = sum(transpose(-y) * log(h) - transpose(1 - y)*log(1 - h)) / m;
tmp = theta;
tmp(1) = 0;
J = J + (lambda*sum(tmp .^ 2))/(2*m);
grad = transpose(X)*(h - y)/m;
grad = grad + (lambda/m).*tmp;


%{
    for i = 1:rows(X)
    h = sigmoid(transpose(theta) * transpose(X(i,:)));
    J = J + ( -y(i)*log(h) - (1-y(i))*log(1 - h)  );
endfor
J = J / m;
J = J + (lambda*sum(theta(2:length(theta)) .^ 2))/(2*m);

for j = 1:rows(theta)
    s = 0;
    for i = 1:rows(X)
        h = sigmoid(transpose(theta) * transpose(X(i,:)));
        s = s + (h - y(i))*X(i,j);
    endfor
    grad(j) = s/m;
    if (j > 1)
        grad(j) = grad(j) + (lambda/m)*theta(j);
    endif

endfor
%}




% =============================================================

end
