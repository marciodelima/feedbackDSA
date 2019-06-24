# Projeto 1
# MARCIO DE LIMA - FORMACAO FCD 2.0
#
# Em resumo, neste projeto, você deverá construir um modelo de aprendizado de máquina para determinar se um clique é fraudulento ou não.
#
# https://www.kaggle.com/c/talkingdata-adtracking-fraud-detection/data
#
# train_sample.csv - 100,000 randomly-selected rows of training data, to inspect data before downloading full set
# #########################
# DICIONARIO DE DADOS
# Each row of the training data contains a click record, with the following features.
# ip: ip address of click.
# app: app id for marketing.
# device: device type id of user mobile phone (e.g., iphone 6 plus, iphone 7, huawei mate 7, etc.)
# os: os version id of user mobile phone
# channel: channel id of mobile ad publisher
# click_time: timestamp of click (UTC)
# attributed_time: if user download the app for after clicking an ad, this is the time of the app download
# is_attributed: the target that is to be predicted, indicating the app was downloaded
# Note that ip, app, device, os, and channel are encoded.
############################
#
#Baseado no problema de negocio informado acima, será criado um modelo do tipo de Classificacao de Machine Learning 
#de Aprendizado Supervisionado.
#

#Setando Diretorio
setwd("~/Cursos_DSA/FCD/BigData_R_Azure/FeedBack/Projeto_Propaganda_Mobile")

#Importando funções e bibliotecas
library(data.table)
library(caret)
library(dplyr)
library(ggplot2)
library(caTools)
library(randomForest)
library(e1071)
library(ROSE)
library(rpart)
library(ROCR)

source("Utils.R")

#Carregando os dados
df <- fread("dataset/train_sample.csv", header = T, sep = ",", stringsAsFactors = FALSE)
str(df)
View(df)

#Data Muning
#Arrumando os dados
colunasFator <- c("is_attributed")
df <- to.factors(df, colunasFator)
df$click_time <- get_asPOSIXct(df, 6)
df$attributed_time <- get_asPOSIXct(df, 7)

#Data Muning
#Limpando a coluna ID, sem utilidade. 
#Obs.: Poderiamos criar faixas pelo IP quebrando numa nova coluna de PAIS de origem, mas fica pro futuro. 
df$ip <- NULL

#Data Muning
#Tratamento dos Campos NA's. Decisão de colocar a mesma Data do click_time
df$attributed_time <- ifelse(is.na(df$attributed_time), df$click_time, df$attributed_time)
df$attributed_time <- as.POSIXct(as.integer(df$attributed_time), origin = "1970-01-01")       

#Balanceamento da Variavel TARGET
table(df$is_attributed)
#Variavel TARGET sem balanceamento, modelo sera tendencioso dessa forma

#Balanceando os dados
df2 = ovun.sample(is_attributed ~ . , data = df, method = "both", p=0.5)$data
table(df2$is_attributed)
View(df2)

#Mapeando as melhores variaveis para o modelo preditivo
modeloSel <- randomForest(is_attributed ~ . , 
                          data = df2, 
                          ntree = 15, 
                          nodesize = 10,
                          importance = TRUE)
varImpPlot(modeloSel)

#Decisao de modelar com as variaveis: "app","channel", "os"

#Split de dados
# Funcao para gerar dados de treino e dados de teste
indice = sample.split(df2, SplitRatio = 0.7)

# Gerando dados de treino e de teste - Separando os dados
dados_treino <- df2[indice==TRUE,]
dados_teste <- df2[indice==FALSE,]
class1 <- dados_teste$is_attributed
dados_teste$is_attributed <- NULL

#Verificando o balanceamento
table(dados_treino$is_attributed)

#Construindo o modelo
formula.init <- "is_attributed  ~ app + channel + os" 
formula.init <- as.formula(formula.init)
formula.full <- "is_attributed  ~ ." 
formula.full <- as.formula(formula.full)

?svm
modelo_svm_v1 <- svm(formula.init , 
                     data = dados_treino, 
                     type = 'C-classification', 
                     kernel = 'radial') 

# Visualizando o modelo
summary(modelo_svm_v1)

# Testando o modelo nos dados de teste
lr.predictions <- predict(modelo_svm_v1, dados_teste, type="response")

# Avaliando o modelo -> 85% de Acuraria
confusionMatrix(table(data = lr.predictions, reference = class1), positive = '1')

#Otimizando o modelo - Teste1
modelo_rf_v1 = rpart(formula.init, 
                     data = dados_treino, control = rpart.control(cp = .0005)) 


class2 <- predict(modelo_rf_v1, dados_teste, type='class')

# Avaliando o modelo -> 97% de Acuraria => Muito Bom
confusionMatrix(class2, class1)

#Otimizando o modelo - Teste2
lr.model <- glm(formula = formula.init, data = dados_treino, family = "binomial")

# Visualizando o modelo
summary(lr.model)

lr.predictions <- predict(lr.model, dados_teste, type="response")
lr.predictions <- round(lr.predictions)

# Avaliando o modelo -> 72% de Acuraria => Razoavel
confusionMatrix(table(data = lr.predictions, reference = class1), positive = '1')

#Otimizando o modelo - Teste3
modelo_rf_v3 = rpart(formula.full, 
                     data = dados_treino, control = rpart.control(cp = .0005)) 


class3 <- predict(modelo_rf_v3, dados_teste, type='class')

# Avaliando o modelo -> 99% de Acuraria => Melhor modelo
confusionMatrix(class3, class1)

# Tabela Comparativa
# 
# * svm com kernel polynomial           => 62%
# * glm com binomial                    => 72%
# * svm com kernel radial               => 85%
# * rpart com as variáveis selecionadas => 97%
# * rpart com todas as variáveis        => 99%
# 
# 
# Muito Obrigado. 
# 
# Att.
# 
# Marcio de Lima
# 
