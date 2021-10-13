# Install some R packages
# =======================
# tidyr: convert data from wide to long format
# ez: compute ANOVAs
# ggplot2: plot figures
# afex, phia: to obtain similar ANOVA output as in SPSS
# gridExtra: allows multiple ggplots
# BayesFactor: compute BF

#code by vikram b baliga
#specify the packages of interest
packages = c("tidyverse","ez","readxl","afex","phia","gridExtra", "BayesFactor","DescTools",
             'knitr','kableExtra')
#use this function to check if each package is on the local machine
#if a package is installed, it will be loaded
#if any are not, the missing package(s) will be installed and loaded
package.check <- lapply(packages, FUN = function(x) {
  if (!require(x, character.only = TRUE)) {
    install.packages(x, dependencies = TRUE)
    library(x, character.only = TRUE)
  }
})
#verify they are loaded
# search()

# compute CI for mean
source('CI_one_sample_from_descr.R')
source('CI_one_sample_for_matrix.R')

# https://doi.org/10.17045/sthlmuni.4981154.v3
source('BF_U.R')

ci = function(data){return(t.test(data)$conf.int[1:2])}


# dm = Dtmp
# lbl = "d´"
# dvy = "Sensitivity (d´)"
plotoverlay = function(dm, lbl = "label", dvy = "Mean amp (µV)", myylim = c(-Inf,Inf)){
  dmg = gather(dm, iv, dv, colnames(dm[,-1]))
  dmg$fp = factor(dmg$fp)
  dmg$iv = factor(dmg$iv, levels=colnames(dm[,-1]))

  dm = subset(dm, select = -fp)
  dmdescr = data.frame("Condition" = colnames(dm),
                       "N" = rep(nrow(dm),length(dm)),
                       "Mean" = apply(dm,2,mean),
                       "CI_LL" = apply(dm,2,ci)[1,],
                       "CI_UL" = apply(dm,2,ci)[2,],
                       "SD" = apply(dm,2,sd))
  dmdescr$Condition = factor(dmdescr$Condition, levels=colnames(dm))
  
  bp = ggplot(dmg)  + 
    geom_point(aes(x=iv, y=dv), shape=1, size=1) +
    geom_line(aes(x=iv, y=dv, group = fp), size = 0.5, alpha = 0.5, color = "gray") +
    theme_bw() + # get rid of background
    labs(title = paste0(lbl, " (N = ",nrow(dm),")")) +
    theme(plot.title = element_text(hjust = 0.5)) +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) + 
    labs(x = "Condition", y = dvy) +
    geom_hline(yintercept = 0)  +
    geom_errorbar(data = dmdescr, aes(x = Condition, ymin = CI_LL, ymax = CI_UL), size=1, color="blue", width=.4) +
    geom_path(data=dmdescr, aes(x = Condition, y = Mean, group = 1), linetype = "solid", size=1, color="blue") +
    geom_point(data=dmdescr, aes(x = Condition, y = Mean), size = 3, color="blue") +
    ylim(myylim[1], myylim[2])
  return(bp)  
  #plot(bp) 
  #filename = paste0(lbl,"_amps")
  #ggsave(paste0(filename,'.png'), dpi = 300, width = 7, height = 4)
}

getdiffs_SIV_s1 = function(d, iv){
  d = subset(DSIV_s1, select=c(paste0(iv,"lo"), paste0(iv,"hi")))
  d$lo = d[,1]
  d$hi = d[,2]
  d$"hi-lo" = d[,2] - d[,1]
  return(d)
}

getdiffs_SIV_s2 = function(d, iv){
  d = subset(DSIV_s2, select=c(paste0(iv,"no"), paste0(iv,"lo"), paste0(iv,"hi")))
  d$no = d[,1]
  d$lo = d[,2]
  d$hi = d[,3]
  d$"lo-no" = d[,2] - d[,1]
  d$"hi-no" = d[,3] - d[,1]
  d$"hi-lo" = d[,3] - d[,2]
  return(d)
}
  
getdiffs_SV = function(d, iv){
  d = subset(DSV, select=c(paste0(iv,"no20Hz"), paste0(iv,"lo20Hz"), paste0(iv,"hi20Hz"),
						 paste0(iv,"no40Hz"), paste0(iv,"lo40Hz"), paste0(iv,"hi40Hz"),
						 paste0(iv,"no80Hz"), paste0(iv,"lo80Hz"), paste0(iv,"hi80Hz")))
  d$no20Hz = d[,1]
  d$lo20Hz = d[,2]
  d$hi20Hz = d[,3]
  d$"lo-no20Hz" = d[,2] - d[,1]
  d$"hi-no20Hz" = d[,3] - d[,1]
  d$"hi-lo20Hz" = d[,3] - d[,2]
  
  d$no40Hz = d[,4]
  d$lo40Hz = d[,5]
  d$hi40Hz = d[,6]
  d$"lo-no40Hz" = d[,5] - d[,4]
  d$"hi-no40Hz" = d[,6] - d[,4]
  d$"hi-lo40Hz" = d[,6] - d[,5]

  d$no80Hz = d[,7]
  d$lo80Hz = d[,8]
  d$hi80Hz = d[,9]
  d$"lo-no80Hz" = d[,8] - d[,7]
  d$"hi-no80Hz" = d[,9] - d[,7]
  d$"hi-lo80Hz" = d[,9] - d[,8]
  return(d)
}

tableCIs = function(d){
  RmCI = CI_one_sample_for_matrix(d)
  RmCI = round(RmCI, digits = 3)
  RmCI = data.frame(colnames(d), RmCI, row.names = NULL)
  colnames(RmCI) <- c('Variable', 'Mean', 'LL', 'UL', 'N')
  return(RmCI)
}

contrastBFs = function (d, iv){
  d = getdiffs(d, iv)
  d = d[,c(13:15,19:21,25:27)] 
  Mean = colMeans(d)
  Mean = round(Mean, 4)

  LL = -1
  UL = 1
  BF10_Q1_U = numeric()
  Vlabels = colnames(d)
  for (i in 1:length(Vlabels)) {
    tmp = d[,i]
    meanobtained = mean(tmp)
    semobtained = sd(tmp)/sqrt(length(tmp))
    dfobtained = length(tmp)-1
    BF10_Q1_U[i] = BF_U(LL, UL, meanobtained, semobtained, dfobtained)}
  BF01_Q1_U = 1/BF10_Q1_U
  
  # {r compute BF 0 to +1, echo = rout, include = rout}
  LL = 0
  UL = 1
  BF10_Q2_U = numeric()
  for (i in 1:length(Vlabels)) {
    tmp = d[,i]
    meanobtained = mean(tmp)
    semobtained = sd(tmp)/sqrt(length(tmp))
    dfobtained = length(tmp)-1
    BF10_Q2_U[i] = BF_U(LL, UL, meanobtained, semobtained, dfobtained)}
  BF01_Q2_U = 1/BF10_Q2_U
  
  # {r compute BF 0 to +0.24, echo = rout, include = rout}
  #For amp
  LL = 0
  UL = 0.24
  BF10_Q3_U = numeric()
  for (i in 1:length(Vlabels)) {
    tmp = d[,i]
    meanobtained = mean(tmp)
    semobtained = sd(tmp)/sqrt(length(tmp))
    dfobtained = length(tmp)-1
    BF10_Q3_U[i] = BF_U(LL, UL, meanobtained, semobtained, dfobtained)}
  BF01_Q3_U = 1/BF10_Q3_U
  
  # {r compute BF 0 to +0.37, echo = rout, include = rout}
  # For ITC
  LL = 0
  UL = 0.37
  BF10_Q4_U = numeric()
  for (i in 1:length(Vlabels)) {
    tmp = d[,i]
    meanobtained = mean(tmp)
    semobtained = sd(tmp)/sqrt(length(tmp))
    dfobtained = length(tmp)-1
    BF10_Q4_U[i] = BF_U(LL, UL, meanobtained, semobtained, dfobtained)}
  BF01_Q4_U = 1/BF10_Q4_U
  
    
  # {r output BF, echo = rout}
  RBF = cbind(BF01_Q1_U, BF01_Q2_U, BF01_Q3_U, BF01_Q4_U)
  RBF = round(RBF, digits = 3)
  RBF = data.frame(Vlabels, Mean, RBF)
  row.names(RBF) = NULL
  colnames(RBF)[1] = "Variable"
  return(RBF)
}

BFfitH1 = function (LL = 0, UL = 1, tmp){
  meanobtained = mean(tmp)
  semobtained = sd(tmp)/sqrt(length(tmp))
  dfobtained = length(tmp)-1
  BF01 = 1/BF_U(LL, UL, meanobtained, semobtained, dfobtained)
  return(BF01)
}

# i = 1
# dataf = Dtmp1[[i]]
# mylvls = c('low','high')
computeANOVA = function(i, dataf, mylvls = c('low','high')){
    Dtmp3 <- gather(dataf, key = cond, value = dv, 2:length(dataf)) %>%
      mutate(load = substr(cond, str_length(cond)-5, str_length(cond)-4),
             load = recode_factor(load, 'no' = 'no', 'lo' = 'low', 'hi' = 'high'),
             load = factor(load, levels = mylvls),
             frequency = substr(cond,str_length(cond)-3, str_length(cond)-2),
             frequency = factor(frequency),
             fp = factor(fp),
             cond = NULL)

  # descriptives
  descrDtmp3 = ezStats(Dtmp3, 
                       dv = dv, 
                       wid = fp, 
                       within = c(load, frequency), 
                       type=3)
  
  # CI
  CIs = CI_one_sample_from_descr(descrDtmp3[,4], descrDtmp3[,3], descrDtmp3[,5], .95)
  descrDtmp3$CI_LL = CIs[,1]
  descrDtmp3$CI_UL = CIs[,2]
  rm(CIs)
  
  # bar chart
  bpa = ggplot(descrDtmp3, aes(x=frequency, y=Mean, fill=load)) + 
    geom_bar(stat="identity"
             ,color="black" # add black border to each bar
             ,position=position_dodge()) + # separate bars
    theme_bw() + # get rid of background
    geom_errorbar(position=position_dodge(.9), width=.25, 
                  aes(ymin=CI_LL, ymax=CI_UL)) + 
    scale_fill_manual(
      values = c("#FFFFFF", "#999999", "#CCCCCC", "#000000"),
      name = "Load") +
    labs(title = paste0(Dvars[i], " for load by frequency")) +
    theme(plot.title = element_text(hjust = 0.5)) +
    labs(x = "Frequency") +
    labs(y = Dvars[i]) +
    labs(legend = "Load")
  plot(bpa)
  filename = file.path(dir_fig, paste0("figure_", 
                                       Dvars[i], "_for_load_by_frequency"))
  ggsave(paste0(filename,'.pdf'), width = 6, height = 4.5, dpi = 600)
  
  
  cat("=====================================================================\n")
  cat(paste0('                        ', Dvars[i]))
  cat("\n=====================================================================\n\n\n")
  
  cat("Descriptives\n")
  cat("============\n")
  descrDtmp3$SD = NULL
  descrDtmp3$FLSD = NULL
  print(descrDtmp3, row.names = FALSE)
  cat("\n\n")
  
  #options(warn = -1)
  # turn off to avoid the warning about HF eps from ANOVA summary
  
  cat("Anova\n")
  cat("=====\n")
  myAnova=aov_car(dv ~ load * frequency + Error(fp/(load*frequency)), data = Dtmp3)
  tmpsum = summary(myAnova)
  # This prints each effect:
  # summary(aov(dv ~ frequency * block + Error(fp/(load*frequency)), data = Dtmp3))
  print(tmpsum)
  cat("\n\n")

  cat("BF without vs with interaction\n")
  cat("==============================\n")
  # compute BF01_Interaction
  # BF for a model without interaction (ie only main effects) vs a model with interaction
  # see mixed models here: https://richarddmorey.github.io/BayesFactor/
  # This tests all models:
  # bf = anovaBF(dv ~ load * frequency + fp, data = Dtmp3, whichRandom = "fp",
  #             whichModels = "withmain", iterations = 20000, rscaleFixed = "medium")
  # whichRandom = "fp", this is a random effect
  # whichModels = "withmain" (default) means that the interaction model includes the main effects
  # iterations = 10000 (default) by I use more to reduce the error
  # rscaleFixed = "medium" (default) : r = 1/2 (default in JASP)
  
  niterations = 30000
  # use lmBF to focus on a particular model
  # model without interaction
  bfwout = lmBF(dv ~ load + frequency + fp, data = Dtmp3, whichRandom = "fp",
                iterations = niterations, rscaleFixed = "medium", progress = FALSE)
  # model with interaction
  bfwith = lmBF(dv ~ load + frequency + load:frequency + fp, data = Dtmp3, whichRandom = "fp",
                iterations = niterations, rscaleFixed = "medium", progress = FALSE)
  bf = bfwout/bfwith
  print(bf)
  Fres[i,1] = Dvars[i]  
  GGtable = tmpsum$pval.adjustments
  which = row.names(GGtable)
  Fres[i,2] = GGtable[which=="load:frequency",2]
  bf <- extractBF(bf)
  Fres[i,3] = bf$bf
  Fres[i,4] = bf$error*100
  
  return(Fres)
}
