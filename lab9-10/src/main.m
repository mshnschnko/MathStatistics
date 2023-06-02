pkg load interval

input = csvread("Channel_1_500nm_0.23mm.csv")
eps = 1e-4

input_int = infsup(input - eps, input + eps)

figure
errorbar(mid(input_int), rad(input_int), "b")
title("Data")
xlabel("n")
ylabel("mV")
xlim([0, size(input, 1)])

[tau1, w1, yint1] = DataLinearModel(input, eps)
sum_w1 = sum(w1)

figure
hold on
errorbar(input, eps, "b")
x = [1, size(input, 1)]
plot(x, tau1(1) + tau1(2) .* x, "r")
title("Data simple regression")
xlabel("n")
ylabel("mV")
xlim([1, size(input, 1)])
ylim([input(1) - eps * w1(1), input(end) + eps * w1(end)])

[tau2, w2, yint2] = DataLinearModelZ(input, eps)
sum_w2 = sum(w2)

figure
hold on
errorbar(input, eps, "b")
x = [1, size(input, 1)]
errorbar(input, eps * w2, "y")
plot(x, tau2(1) + tau2(2) .* x, "r")
title("Data regression with reducement of intervals")
xlabel("n")
ylabel("mV")
xlim([1, size(input, 1)])
ylim([input(1) - eps * w2(1), input(end) + eps * w2(end)])

setenv ("OCTAVE_LATEX_DEBUG_FLAG", "1")

alpha = sum_w2 / 100

figure
hold on

plot(w1)
plot(w2)
plot(alpha .* ones(size(w2, 1)), "k--")

title("\\omega_1 and \\omega_2 values")
legend("\\omega_1", "\\omega_2")
ylabel("\\omega_i")
xlabel("i")
text(5, alpha + 0.05, "\\alpha");

ylim([0, max(w2) + 0.5])

x = (1 : size(input, 1)).'

residuals1 = input_int - tau1(1) - tau1(2) .* x

figure
hold on
errorbar(mid(residuals1), rad(residuals1), "b")
plot(x, zeros(size(x, 1)), "r")
title("Regression residuals")
ylabel("mV")
xlabel("n")
xlim([1, size(residuals1, 1)])
s = [inf(residuals1).' sup(residuals1).']
ylim([min(s), max(s)])

x = (1 : size(input, 1)).'
residuals2 = input_int - tau2(1) - tau2(2) .* x

figure
hold on
errorbar(mid(residuals2), rad(residuals2), "b")
plot(x, zeros(size(x, 1)), "r")
title("Regression with reducement residuals")
ylabel("mV")
xlabel("n")
xlim([1, size(residuals2, 1)])
s = [inf(residuals2).' sup(residuals2).']
ylim([min(s), max(s)])

[mode1, modefreq1, freqs1, Ss1] = imodeR([inf(residuals1) sup(residuals1)])
[mode2, modefreq2, freqs2, Ss2] = imodeR([inf(residuals2) sup(residuals2)])

figure
hold on

plot(Ss1(1:end-1), freqs1, "r")
plot(Ss2(1:end-1), freqs2, "b")

title("Mode frequencies of residuals")
xlabel("mV")
ylabel("\\mu")
legend("1st model \\omega \\geq 1", "2nd model \\omega \\geq 0")
xlim([min(Ss1) max(Ss1)])

x = 1:size(input,1).'
[irproblem] = ir_problem([x.^0; x].', input, max(w2) * eps)

vertices = ir_beta2poly(irproblem)

b_int = ir_outer(irproblem)

figure
hold on
x = vertices(:, 1)
y = vertices(:, 2)
ir_plotbeta(irproblem)
ir_plotrect(b_int, "r")
title("Information set")
xlim([min(x) - 1e-5, max(x) + 1e-5])
ylim([min(y) - 1e-7, max(y) + 1e-7])
xlabel("\\beta_0")
ylabel("\\beta_1")

xlimits = [1 size(input, 1)]

figure
hold on
ir_plotmodelset(irproblem, [-50 250])
errorbar(input, max(w2) * eps, "b")

xlim(xlimits)
ylim([input(1) - max(w2) * eps, input(end) + max(w2) * eps])

title("Corridor of joint dependencies")
xlabel("n")
ylabel("mV")

figure
hold on
ir_plotmodelset(irproblem, [-50 250])
errorbar(input, max(w2) * eps, "b")

xlim(xlimits)
ylim([input(1) - max(w2) * eps, input(end) + max(w2) * eps])

title("Corridor of joint dependencies")
xlabel("n")
ylabel("mV")

xlim("auto")
ylim("auto")

