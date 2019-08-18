function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % iter
    HX = X * theta;

    diff = HX - y;

    num_thetas = size(theta)(1,1);
    thetaDeltas = zeros(num_thetas, 1);
    for thetaNum = 1:num_thetas
        % size(diff)
        % size(X(:,thetaNum))
        thetaMult = diff .* X(:,thetaNum);
        thetaDeltas(thetaNum, 1) = (alpha/m) * sum(thetaMult);
    end
    theta = theta - thetaDeltas;

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
