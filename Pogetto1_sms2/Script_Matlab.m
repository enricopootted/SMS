%% Eseguire il codice con il tasto "RUN"
%% Cariamento dataset
clearvars
close all
clc

dataset = readtable('dataset.csv');

%% Conversione in matrici

%Condiseriamo tutti i regressori
X = table2array(dataset(:, 2:6));
X = [ones(length(X), 1) X];
%Preleviamo l'altezza della neve
y = table2array(dataset(:, 7));
a=normpdf(y)
area(a)

%Calcolo al numerosità
N = length(X);
%Calcolo numero di regressori del campione
k = width(X) - 1;
%Calcolo determinante matrice (X'X)
detXX = det(X'*X);

%% Stima del modello di regressione OLS

[y_hat, residui, D_sp, D_res, D_tot, beta_hat, jb] = calcoloModelloOLS(X,y);

%% MSE, coefficiente di determinazione e media dei residui

R_2 = D_sp/D_tot;
MSE = D_res/(N - k - 1);
m_residui = mean(residui);


%% Test d'ipotesi sulla significatività dei regressori

%Prelevo la diagonale della matrice
v = diag(inv(X'*X));
%fisso alfa
alpha = 0.01;
%Calcolo la t critica
tCrit = tinv(1-alpha/2, N - k - 1);

%Eseguo test sui coefficienti 
[ris, t] = testCoefficienti(beta_hat, MSE, v, tCrit, k);

%% Validazione del modello con il calcolo dell'MSE di train

%calcolo dell'MSE con algoritmo k-fold (assegnato K=10) sviluppato dalla squadra
[MSE_test] = kfold(10, N, X, y);

%Grafici presenti nella relazione
figure
histfit(residui)
title('DISTRIBUZIONE DEI RESIDUI')
legend('residui','campana gaussiana')
figure
plot(dataset.Data_Ora,y,'k',dataset.Data_Ora,y_hat,'r')
legend('Y OSSERVATI','Y STIMATI')
xlabel('TEMPO')
ylabel('ALTEZZA NEVE cm')

%stampa dei risultati
disp("det(X'X) = " + detXX + " => stime non scadenti");

disp(" ");
disp("Coefficienti:")
disp(" ");
st = sprintf("Regressore\t\t" + "Valore\t\t\t" + "tStat\t\t\t" + "Significativo");
disp(st);
st = sprintf("Intercetta\t\t" + beta_hat(1) + "\t\t" + t(1) + "\t\t" + ris(1));
disp(st);
st = sprintf("Beta_" + (2) + "\t\t\t" + beta_hat(2) + "\t\t\t" + t(2) + "\t\t" + ris(2));
disp(st);
st = sprintf("Beta_" + (3) + "\t\t\t" + beta_hat(3) + "\t\t" + t(3) + "\t\t\t" + ris(3));
disp(st);
st = sprintf("Beta_" + (4) + "\t\t\t" + beta_hat(4) + "\t\t" + t(4) + "\t\t" + ris(4));
disp(st);
st = sprintf("Beta_" + (5) + "\t\t\t" + beta_hat(5) + "\t\t" + t(5) + "\t\t\t" + ris(5));
disp(st);
st = sprintf("Beta_" + (6) + "\t\t\t" + beta_hat(6) + "\t\t\t" + t(6) + "\t\t\t" + ris(6));
disp(st);

disp(" ");
disp("R^2 = " + R_2);
disp("MSE = " + MSE);
disp("MSE di test = " + MSE_test);

disp(" ");
disp("Media dei residui = " + m_residui + " => stimatori non distorti (non asintoticamente)");
disp("Statistica JB sui residui = " + jb + " => i residui non sono normali");

clearvars -except X y beta_hat R_2 MSE_test ris jb detXX t tCrit MSE m_residui

%% Definizione delle funzioni

function [y_hat, residui, D_sp, D_res, D_tot,beta_hat, jb] = calcoloModelloOLS(X,y)
%CALCOLOMODELLOOLS Calcolo del modello OLS e dei principali indicatori
%   

Xt = X';
beta_hat = (Xt*X)\(Xt*y);

%Calcolo valori trasposti
y_hat = X*beta_hat;

%Devianza totale
m_y = mean(y);
D_tot = (y - m_y)'*(y - m_y);

%calcolo dei residui
residui = y - y_hat;
D_res = residui'*residui;
D_sp = D_tot - D_res;
jb = (length(residui)/6)*(skewness(residui)^2) + (length(residui)/24)*((kurtosis(residui)-3)^2);

end

function [MSE_test] = kfold(K,N,X,y)
%KFOLD Calcolo dell'MSE_test con algoritmo K-fold e K assegnabile
%

B = randperm(N);
X_p = X(B, :);
y_p = y(B);

%Stabilisco la dimensione dei fold
n = N/K;

%itero l'algoritmo per calcolare MSE di test su ogni fold
MSE_test = 0;
for i = 1:K
    %Costruzione dati di training
    Xk = [X_p(1:(n*(i-1)+1), :); X_p((n*i):end, :)];
    yk = [y_p(1:(n*(i-1)+1), :); y_p((n*i):end, :)];
    
    %Costruzione dati di test
    fold = X_p((n*(i-1)+1):(n*i), :);
    y_fold = y_p((n*(i-1)+1):(n*i), :);
    
    %Stima modello con dati di training
    B_CV = (Xk'*Xk)\(Xk'*yk);
    
    %calcolo dei residui e stima della varianza
    res_CV = y_fold - fold*B_CV;
    
    MSE_test = MSE_test + (res_CV'*res_CV)/length(y_fold);
end

%media della varianza di test
MSE_test = MSE_test/K;

end

function [ris, t] = testCoefficienti(beta_hat,MSE_train,v,tCrit, k)
%TESTCOEFFICIENTI Test sui singoli coefficienti della regressione,
%sfruttando la t di student
%

ris = zeros(6,1);
t = zeros(k+1, 1);
for i = 1:(k+1)
    %calcolo delle statistiche test
    t(i) = beta_hat(i)/sqrt(MSE_train*v(i));
    
    %nella matrice ris si ottiene:
    if t(i)>(-tCrit) && t(i)<tCrit
        ris(i) = 0;     %0 se il coefficiente non è significativo
    else
        ris(i) = 1;     %1 se il coefficiente è significativo
    end
end

end


