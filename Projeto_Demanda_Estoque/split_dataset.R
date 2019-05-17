# Carregando bibliotecas
library(data.table)
library(dplyr)

#Setando o Diretorio 
setwd("~/Cursos_DSA/FCD/BigData_R_Azure/FeedBack/Projeto_Demanda_Estoque")

#Carregando os dados
df <- fread("dataset/train.csv", header = T, sep = ",", stringsAsFactors = FALSE)
#Filtrando valores e escolhendo aleatoriamento 2 milhoes de registros
df2 <- df %>% filter(Venta_uni_hoy > 0) %>% sample_n(2000000)
#Escrevendo o resultado num .csv apartado
write.csv2(df2, file="dataset/train_sample.csv",sep = ",") 

#Gerando o Mini Train
df <- fread("dataset/train_sample.csv", header = T, sep = ";", stringsAsFactors = FALSE)
#Filtrando valores e escolhendo aleatoriamento 100.000 registros
df2 <- df %>% sample_n(100000)
#Escrevendo o resultado num .csv apartado
write.csv2(df2, file="dataset/train_mini.csv") 
