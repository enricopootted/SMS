clear all
clc
%%LAVORO DI:
%%Corrini Francesco
%%Imberti Federico
%%Ingoglia Andrea
%%Perani Enrico

%%Caricamento dati
dataset = readtable('dataset.csv');

%%Rinominazione delle variabili, scompattamento della data
variabili = {'Temperatura','umidit_Relativa_','RadiazioneGlobale','Velocit_Vento','AltezzaNeve'};
dataset.Formato_data = datenum(dataset.Data);
variabili = ['Formato_data', variabili];
dataset.anno = year(dataset.Data);
dataset.mese = month(dataset.Data);
dataset.giorno = day(dataset.Data);

%%Raggruppamento dei dati in base al giorno, divisione in training e
%%forecast
dati_giornalieri = grpstats(dataset, {'anno','mese','giorno'}, 'mean', 'DataVars', variabili ,'VarNames', [{'anno', 'mese', 'giorno', 'GroupCount'}, variabili]);
pioggia_cumulata = grpstats(dataset, {'anno','mese','giorno'}, 'sum', 'DataVars', 'precipitazioneMm', 'VarNames', [{'anno', 'mese', 'giorno', 'GroupCount'}, 'precipitazioneMm']);
dati_giornalieri = addvars(dati_giornalieri, pioggia_cumulata.precipitazioneMm, 'Before', 'AltezzaNeve', 'NewVariableNames', 'precipitazioneMm');
dati_giornalieri = addvars(dati_giornalieri, datetime(datevec(dati_giornalieri.Formato_data)), 'Before', 'anno', 'NewVariableNames', 'Data');

dati_giornalieri_forecast = dati_giornalieri(362:end, :);
dati_giornalieri = dati_giornalieri(1:362, :);

%%conversione in matrici
X = table2array(dati_giornalieri(:,7:11)); 
y = table2array(dati_giornalieri(:, 12));
XForecast = table2array(dati_giornalieri_forecast(:,7:11));
yForecast = table2array(dati_giornalieri_forecast(:,12));

%analisi stazionarietà delle covariate e della variabile risposta
for i = 1:width(X)
    h_adf_X(i) = adftest(X(:, i)); %tutte stazionarie
end

h_adf_y = adftest(y); %non stazionaria, si calcola la differenza prima

%calcolo delle differenze prime
X = diff(X);
y = diff(y);
XForecast = diff(XForecast);
yForecast = diff(yForecast);

%calcolo ols e analisi dei residui
Xone = X;
Xone(:,6) = 1;
beta = regress(y, Xone);

resOLS = y - Xone*beta;
[h_lbq_resOLS, p_lbq_resOLS] = lbqtest(resOLS); %residui correlati                         
[h_adf_resOLS, p_adf_resOLS] = adftest(resOLS); %residui stazionari  

%Funzioni di autocorrelazione e correlazione parziale dei residui
%Grafici delle due funzioni
autocorr(resOLS)
parcorr(resOLS)

%Algoritmo per la ricerca del miglior arima in base a AIC e BIC
pMAX = 4; %massimo autoregressivo
qMAX = 4; %massima media mobile

AIC = zeros(pMAX,qMAX); %salvo AIC e BIC di ogni modello stimato
BIC = zeros(pMAX,qMAX);

for p = 1:pMAX
    for q = 1:qMAX
        modello = regARIMA(p,0,q);
        stima_modello = estimate(modello, y, 'X', X, 'display', 'off');
        summ = summarize(stima_modello);
        AIC(p,q) = summ.AIC;
        BIC(p,q) = summ.BIC;
    end
end

AIC_minimo = min(min(AIC));             
[p_AIC,q_AIC] = find(AIC == AIC_minimo);
BIC_minimo = min(min(BIC));
[p_BIC,q_BIC] = find(BIC == BIC_minimo);

%AIC e BIC dei due modelli molto simili
%Scelto quello più parsimonioso, ARMA(1,1)

%Stima del modello
modello = regARIMA(p_BIC,0,q_BIC);
stima_modello = estimate(modello, y, 'X', X, 'display', 'off');

%Diagnostica dei residui
resARMA = infer(stima_modello, y, 'X', X);

[h_jb_resARMA, p_jb_resrARMA] = jbtest(resARMA); %residui non gaussiani                         
[h_lbq_resARMA, p_lbq_resARMA] = lbqtest(resARMA); %residui incorrelati                         
[h_adf_resARMA, p_adf_resARMA] = adftest(resARMA); %residui stazionari

%Grafico dei residui (che sono whitenoise) + hisftit per vedere che non
%sono normali
    
%Dato che i residui non sono gaussiani facciamo inferenza tramite bootstrap

N = 20; %numero di simulazioni

beta = zeros(N, 5);           %Matrice dei coefficienti stimati ad ogni iterazione
k = 3;                        %passi a cui vogliamo la previsione

for i = 1:N
    campione = randsample(resARMA, length(X), true);
    y_simulata = filter(stima_modello, campione, 'X', X);
    
    modello_simulato = estimate(modello, y_simulata, 'X', X, 'Display', 'off');
    x = modello_simulato.Beta;
    beta(i, :) = x;
end

%Stime forecast
alpha = 0.05;

%IC dei coefficienti beta
beta_bootstrap = mean(beta);
IC_beta = [quantile(beta, alpha/2); quantile(beta, 1-alpha/2)];

%Inserire 5 grafici delle distribuzioni dei coefficienti con gli intervalli
%di confidenza

%Dato che non abbiamo riscontrato significatività nei coefficienti
%stiamiamo il modello come un arima

%Algoritmo per la ricerca del miglior arima in base a AIC e BIC
pMAX = 4; %massimo autoregressivo
qMAX = 4; %massima media mobile

AIC = zeros(pMAX,qMAX); %salvo AIC e BIC di ogni modello stimato
BIC = zeros(pMAX,qMAX);

for p = 1:pMAX
    for q = 1:qMAX
        modello = arima(p,0,q);
        stima_modello = estimate(modello, y, 'display', 'off');
        summ = summarize(stima_modello);
        BIC(p,q) = summ.BIC;
    end
end

BIC_minimo = min(min(BIC));
[p_BIC, q_BIC] = find(BIC == BIC_minimo);

%Stima del modello
modello = arima(p_BIC,0,p_BIC);
stima_modello = estimate(modello, y);

%Analisi dei residui
resARMA = infer(stima_modello, y);

[h_jb_resARMA, p_jb_resARMA] = jbtest(resARMA); %residui non gaussiani                         
[h_lbq_resARMA, p_lbq_resARMA] = lbqtest(resARMA); %residui incorrelati                         
[h_adf_resARMA, p_adf_resARMA] = adftest(resARMA); %residui stazionari  
%Grafico dei residui (che sono whitenoise) + hisftit per vedere che non
%sono normali

[yPrevista, Var_y] = forecast(stima_modello, 3,'Y0', y);
IC_previsioni = [yPrevista - norminv(1-alpha/2)*sqrt(Var_y), yPrevista + norminv(1-alpha/2)*sqrt(Var_y)];
%Grafico delle previsioni (con IC) sopra a quello delle vere y

clearvars -except beta_bootstrap dati_giornalieri dati_giornalieri_forecast IC_beta stima_modello p_BIC q_BIC resARMA yPrevista Var_y h_jb_resARMA h_lbq_resARMA h_adf_resARMA IC_previsioni