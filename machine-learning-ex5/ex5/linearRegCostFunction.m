function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the
%   cost of using theta as the parameter for linear regression to fit the
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

num_thetas = size(theta)(1,1);

HX = X * theta;

diff = HX - y;

diff = diff .^ 2;

J = (1/(2 * m)) * sum(diff);

% Now adding the lambda part to this.
thetaSquareSum = (lambda/(2 * m)) * sum(theta(2:num_thetas) .^ 2);

J = J + thetaSquareSum;

% Computing the grad.
diff = HX - y;
grad = (1/m) * (X' * diff);

% Now adding the regularization to the grad ...
multiplier = ones(num_thetas, 1);
multiplier(1, 1) = 0;

grad = grad + (lambda/m) .* multiplier .* theta;










% =========================================================================

grad = grad(:);

end
