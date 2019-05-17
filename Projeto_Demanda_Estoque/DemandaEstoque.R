# Projeto 2
# MARCIO DE LIMA - FORMACAO FCD 2.0
#
# Em resumo, Neste projeto de aprendizado de máquina, você deve desenvolver um
#modelo para prever com precisão a demanda de estoque com base nos dados
#históricos de vendas.
#
# https://www.kaggle.com/c/grupo-bimbo-inventory-demand
#
#The dataset you are given consists of 9 weeks of sales transactions in Mexico. 
#Every week, there are delivery trucks that deliver products to the vendors. 
#Each transaction consists of sales and returns. Returns are the products that are unsold and expired. 
#The demand for a product in a certain week is defined as the sales this week subtracted by the return next week.
#
# Observações:   
# * There may be products in the test set that don't exist in the train set. This is the expected behavior of inventory data, since there are new products being sold all the time. Your model should be able to accommodate this.
# * There are duplicate Cliente_ID's in cliente_tabla, which means one Cliente_ID may have multiple NombreCliente that are very similar. This is due to the NombreCliente being noisy and not standardized in the raw data, so it is up to you to decide how to clean up and use this information. 
# * The adjusted demand (Demanda_uni_equil) is always >= 0 since demand should be either 0 or a positive value. The reason that Venta_uni_hoy - Dev_uni_proxima sometimes has negative values is that the returns records sometimes carry over a few weeks.
#
#The train and test dataset are split based on time, as well as the public and private leaderboard dataset split.
#
# #########################
# DICIONARIO DE DADOS
# * Semana — Week number (From Thursday to Wednesday)
# * Agencia_ID — Sales Depot ID
# * Canal_ID — Sales Channel ID
# * Ruta_SAK — Route ID (Several routes = Sales Depot)
# * Cliente_ID — Client ID
# * NombreCliente — Client name
# * Producto_ID — Product ID
# * NombreProducto — Product Name
# * Venta_uni_hoy — Sales unit this week (integer) 
# * Venta_hoy — Sales this week (unit: pesos)
# * Dev_uni_proxima — Returns unit next week (integer)
# * Dev_proxima — Returns next week (unit: pesos)
# * Demanda_uni_equil — Adjusted Demand (integer) (This is the target you will predict)
#
# Baseado no problema de negocio informado acima, será criado um modelo do tipo de Regressão de Machine Learning 
# de Aprendizado Supervisionado.
#
# OBS:
# COMO O DATASET É GIGANTE, MAIS DE 3GB COM 74 MILHOES DE OBSERVACOES, 
# DECIDI, GERAR UM DATASET MENOR COM 2 MILHOES DE OBS ESCOLHIDAS ALEATORIAMENTE PARA A ANALISE EXPLORATORIA
# CRIADO UM MINI CSV COM 100.000 LINHAS PARA A MONTAGEM DO MODELO PREDITIVO
# JA RETIRANDO DADOS NA, OS FONTES DOS SPLITS ESTAO CONTIDOS NO ARQUIVO split_dataset.R 
#
#Setando Diretorio
setwd("~/Cursos_DSA/FCD/BigData_R_Azure/FeedBack/Projeto_Demanda_Estoque")

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

source("Utils.R")

#Carregando os dados - Dados de exemplo - 2 Milhoes de obs
df <- fread("dataset/train_sample.csv", header = T, sep = ";", stringsAsFactors = FALSE)
df2Mini <- fread("dataset/train_mini.csv", header = T, sep = ";", stringsAsFactors = FALSE)
str(df)
View(df)

#Analise Exploratoria dos dados
produtos <- sample(unique(df[,Producto_ID]),10)
list_produtos <- df[Producto_ID %in% produtos]

#Valores positivos da variavel target
list_produtos[,Match:=(Venta_uni_hoy-Dev_uni_proxima)==Demanda_uni_equil]

list_produtos

#Gerando um grafico para visualizar  e explorar melhor os dados
list_produtos %>% 
  ggplot(aes(x=Demanda_uni_equil))+
  geom_histogram(binwidth=2)+
  facet_wrap(~Producto_ID)+
  xlim(c(0,100)) +
  scale_x_continuous(name="")+
  scale_y_continuous(name="")+
  ggtitle("Quantidade de Demanda por Produtos")

#Gerando um grafico para visualizar e explorar melhor os dados
list_produtos %>% 
  ggplot(aes(x=Semana,y=Demanda_uni_equil)) + 
  geom_point(aes(color=factor(Canal_ID))) + 
  facet_wrap(~Producto_ID,scale="free_y") + 
  scale_x_continuous(name="Semana")+
  scale_y_continuous(name="")+
  ggtitle("Demandas de Produto por Semana e Canal")

#Data Muning
#Limpeza dos dados
df2Mini$V1 <- NULL
df2Mini$Cliente_ID <- NULL
df2Mini$Ruta_SAK <- NULL
df2Mini$Agencia_ID <- NULL

View(df2Mini)

#Arrumando os dados
colunasFator <- c("Semana", "Canal_ID")
df2Mini <- to.factors(df2Mini, colunasFator)

#Arrumando Tipagem das Variaveis
df2Mini$Venta_hoy = as.double(sub(",", ".", df2Mini$Venta_hoy))
df2Mini$Dev_proxima = as.double(sub(",", ".", df2Mini$Dev_proxima)) 

#Criando nova coluna com o formula (Venta_uni_hoy - Dev_uni_proxima)
df2Mini <- mutate(df2Mini, quantidade=Venta_uni_hoy - Dev_uni_proxima)

View(df2Mini)
str(df2Mini)

#Mais Limpeza dos dados, deixando no Data Frame, somente as colunas que fazem mais sentido
df2Mini$Venta_uni_hoy <- NULL
df2Mini$Venta_hoy <- NULL
df2Mini$Dev_uni_proxima <- NULL
df2Mini$Dev_proxima <- NULL

#Mapeando as melhores variaveis para o modelo preditivo, confirmando as variaveis
modeloSel <- randomForest(Demanda_uni_equil ~ . , 
                          data = df2Mini, 
                          ntree = 15, 
                          nodesize = 10,
                          importance = TRUE)
varImpPlot(modeloSel)

#Split de dados
# Funcao para gerar dados de treino e dados de teste
indice = sample.split(df2Mini, SplitRatio = 0.7)

# Gerando dados de treino e de teste - Separando os dados
dados_treino <- df2Mini[indice==TRUE,]
dados_teste <- df2Mini[indice==FALSE,]
class1 <- dados_teste$Demanda_uni_equil
dados_teste$Demanda_uni_equil <- NULL

#Construindo o modelo - Primeiro modelo - Stochastic Gradient Boosting
modFit <- train(Demanda_uni_equil ~ .,method="gbm",data=dados_treino, verbose=FALSE)
modFit

# Testando o modelo com os dados de teste
lr.predictions <- round(predict(modFit, dados_teste, type="raw"))

#Avaliação => 94%
mean(class1 == lr.predictions) 

#Otimizando o modelo - Segundo modelo - Regressão Linear
modFit2 = lm(Demanda_uni_equil ~ ., 
                data = dados_treino) 

lr.predictions2 <- round(predict(modFit2, dados_teste, type="response"))

#Avaliação => 99%
mean(class1 == lr.predictions2) 

# Segundo modelo foi o melhor

# Tabela Comparativa
# 
# * gbm - Primeiro Modelo           => 94%
# * lm - Segundo Modelo             => 99%
# 
# Muito Obrigado. 
# 
# Att.
# 
# Marcio de Lima
# 
