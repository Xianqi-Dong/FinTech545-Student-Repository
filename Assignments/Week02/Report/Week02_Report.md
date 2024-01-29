# Xianqi Dong

# Week02

## Problem 1

### a.

$$
\hat{\mu_1} = E[X] = \frac{1}{n}\sum_{i}^{n}x_i = 1.0489703904839582 \\ 
\hat{\mu_2} = E[(X-\hat{\mu_1})^2] = \frac{1}{n-1}\sum_{i}^{n}(x_i-\hat{\mu_1})^2 = 5.4272206818817255 \\ 
\sigma = \frac{1}{n}\sum_{i}^{n}(x_i-\hat{\mu_1})^2 = 5.421793461199844 \\ 
\hat{\mu_3} = E[(\frac{X-\hat{\mu_1}}{\sigma})^3] = \frac{1}{n}\sum_{i}^{n}(\frac{X-\hat{\mu_1}}{\sigma})^3 = 0.8806086425277379 \\
\hat{\mu_4} = E[(\frac{X-\hat{\mu_1}}{\sigma})^4] - 3 = \frac{1}{n}\sum_{i}^{n}(\frac{X-\hat{\mu_1}}{\sigma})^4 - 3 = 23.122200789989723
$$

### b.

Mean: 1.048970

Var: 5.427221

Skew: 0.881932

Kurt: 23.244253

### c.

Simulate a list of 100 data and use formular and package to calculate the statistics. Repeat this process 1000 times. Mean and Var have no difference. So these are unbiased. Use Student T to test whether Skew and Kurt are biased or not. Hypotheses: 




$$
H_0: Skew = \hat{\mu_3} \\ 
H_1: Skew \neq \hat{\mu_3} \\ 
t_{Skew} = \frac{Skew - \hat{\mu_3}}{\sqrt{Var(Skew)/n}} \\

H_0: Kurt = \hat{\mu_4} \\ 
H_1: Kurt \neq \hat{\mu_4} \\ 
t_{Kurt} = \frac{Kurt - \hat{\mu_4}}{\sqrt{Var(Kurt)/n}} \\
$$

Result: 

------------------------------------
1st Moment:                 0.495990

2nd Moment:                 0.082549

3th Moment:                 0.020703

4th Moment:                -1.120182


------------------------------------
Mean:                       0.495990

Var:                        0.082549

Skew:                       0.021809

Kurt:                      -1.101909


------------------------------------
t_skew:                     0.130940

p_skew:                     0.895849

t_kurt:                     1.903562

p_kurt:                     0.057254

------------------------------------

if $\alpha = 0.05, p>\alpha$, so we can't refuse $H_0$. 

## Problem 2

### a.

OLS Beta:   [-0.08738446427005078, 0.7752740987226112]
OLS std:  1.003756319417732

MLE Beta:  [-0.0873844781126185, 0.7752740776445967]
MLE std:  1.003756309651628
R^2:  0.3456068835648125

$\beta_{MLE}$ is a little smaller than $\beta_{OLS}$ because minimum optimization. MLE's standard deviation is greater than OLS's.

Respectively, for 2 methods calculate:
$$
\epsilon = \hat{\beta_1}\bar{X} + \hat{\beta_0} - \bar{Y} \\ 
\epsilon_{OLS} = 0.36300931774742706 \\ 
\epsilon_{MLE} = 0.3630092926962777
$$
OLS's expected residual is also greater than MLE's.

### b.

MLE_t Beta:  [-0.0972694029223096, 0.6750091823157014]
MLE_t std:  7.159786463810651
R^2:  0.3396547079914969

MLE under normality assumption is the best of fit.

### c.

<img src="C:\Users\11833\Documents\fintech545\FinTech545-Student-Repository\Week02\Report\Problem2_value.png" alt="Problem3_pacf" style="zoom:80%;" />
$$
X_2=\hat{\beta}X_{1obs} + \hat{\beta_0}
$$
<img src="C:\Users\11833\Documents\fintech545\FinTech545-Student-Repository\Week02\Report\Problem2_x1pdf.png" alt="Problem2_x1pdf" style="zoom:80%;" />

<img src="C:\Users\11833\Documents\fintech545\FinTech545-Student-Repository\Week02\Report\Problem2_x2pdf.png" alt="Problem2_x2pdf" style="zoom:80%;" />

$X_2$ has a linear relation with observed $X_1$. $X_1$ has a normal distribution. So $X_2$ will have the same type of distribution with $X_1$.

### d.

$$
l=\prod_{i=1}^{n}f(Y|X;\beta) \\ 
ll=\sum_{i=1}^{n}ln(f(Y|X;\beta)) \\ 
ll=\sum_{i=1}^{n}ln(f(x_i))=\sum_{i=1}^{n}[-\frac{1}{2}ln(\sigma^22\pi)-\frac{1}{2}(\frac{x_i-\mu}{\sigma})^2] \\ 
=-\frac{n}{2}ln(\sigma^22\pi)-\frac{1}{2\sigma^2}\sum_{i=1}^{n}(x_i-\mu)^2 \\ 
=-\frac{n}{2}ln(\sigma^22\pi)-\frac{1}{2\sigma^2}(Y-X\hat{\beta})(Y-X\hat{\beta})' \\ 
\frac{\Delta ll}{\Delta \hat{\beta}} = -\frac{1}{\sigma^2}X'(Y-X\hat{\beta})  = 0 \\ 
\therefore \hat{\beta} = (X'X)^{-1}X'Y
$$



## Problem 3

<img src="C:\Users\11833\Documents\fintech545\FinTech545-Student-Repository\Week02\Report\Problem3_pacf.png" alt="Problem3_pacf" style="zoom:80%;" />

<img src="C:\Users\11833\Documents\fintech545\FinTech545-Student-Repository\Week02\Report\Problem3_acf.png" alt="Problem3_acf" style="zoom:80%;" />![Problem3_residual](C:\Users\11833\Documents\fintech545\FinTech545-Student-Repository\Week02\Report\Problem3_residual.png)

​                  AIC                  R^2

AR1    1644.655505     0.042621

AR3    1436.659807     0.373768

MA1    1891.667876    -0.591070

MA3    2788.209192    -8.778802

According to plots and information above, AR(3) is best of fit.
