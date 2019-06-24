# Projeto 1
# MARCIO DE LIMA - FORMACAO FCD 2.0
# CLASSE DE FUNCOES UTILITARIAS
#
##########################################################

## Transformar variaveis em fator
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

## Normalização
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center=T, scale=T)
  }
  return(df)
}

##Transformar coluna em Datetime
get_asPOSIXct <- function(data, ncoluna) {
  novaColuna <-
  as.POSIXct(strptime(
    as.character(data[[ncoluna]]), 
    "%Y-%m-%d %H:%M:%S"))
}

# Biblioteca para geração de curvas ROC
plot.roc.curve <- function(predictions, title.text){
  perf <- performance(predictions, "tpr", "fpr")
  plot(perf, col = "black", lty = 1, lwd = 2, 
       main = title.text, cex.main = 0.6, 
       cex.lab = 0.8, xaxs="i", yaxs="i")
  abline(0,1, col = "red")
  auc <- performance(predictions, "auc")
  auc <- unlist(slot(auc, "y.values"))
  auc <- round(auc,2)
  legend(0.4,0.4, legend = c(paste0("AUC: ",auc)), cex = 0.6, bty = "n", box.col = "white")
  
}

plot.pr.curve <- function(predictions, title.text){
  perf <- performance(predictions, "prec", "rec")
  plot(perf, col = "black", lty = 1, lwd = 2,
       main = title.text, cex.main = 0.6, cex.lab = 0.8, xaxs = "i", yaxs = "i")
}
