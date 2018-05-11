function [err, theta] = LassoADMM(A, b, lda, rho, abs_tolerance, rel_tolerance, iters)
m = size(A, 1);
n = size(A, 2);

L = chol(A'*A + rho*speye(n), 'lower');
U = L';
ATb = A'*b;

primal_residuals = [];
dual_residuals = [];
err_lasso = [];
%spy(L)
x_ad = zeros(n, 1);
y_ad = zeros(n, 1);
z = zeros(n, 1);

err_lasso = [err_lasso; (1/(2*m))*sum((A*x_ad - b).^2)];

for i = 1:iters
%    i
    q = ATb + rho*(z - y_ad);
    x_ad = U \ (L \ q);
    z_prev = z;
    xy = x_ad + y_ad;
    z = max(0, xy - lda/rho) - max(0, -xy - lda/rho);
    y_ad = xy - z;
    primal_residual = norm(x_ad - z);
    primal_residuals = [primal_residuals; primal_residual];
    dual_residual = norm(-rho*(z - z_prev));
    dual_residuals = [dual_residuals; dual_residual];
    epi_primal = sqrt(n)*abs_tolerance + rel_tolerance*max(norm(x_ad), norm(-z));
    epi_dual = sqrt(n)*abs_tolerance + rel_tolerance*norm(y_ad);
    err_lasso = [err_lasso; (1/(2*m))*sum((A*x_ad - b).^2)];
    if primal_residual < epi_primal && dual_residual < epi_dual
        break;
    end
end

figure;
plot(primal_residuals);
hold on;
title('Primal Residual for rho = 10');
ylabel('Primal Residual');
xlabel('Iteration');

figure;
plot(dual_residuals);
hold on;
title('Dual Residual for rho = 10');
ylabel('Dual Residual');
xlabel('Iteration');

iterations = (0:i)';
figure;
plot(iterations, err_lasso);
hold on;
xlabel('Number of Iterations');
ylabel('Squared Error');
title('Convergence with number of Iterations - ADMM Lasso Regression');
err = err_lasso(size(err_lasso, 1));
theta = x_ad;
end