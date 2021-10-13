BF_U<-function(LL, UL, meanobtained, semobtained, dfobtained)
# similar to BF_t (see there for more info) but the H1 is modelled as a uniform.
# LL = lower limit of uniform
# UL = upper limit of uniform
#  
# Computes the BayesFactor(H1 vs H0) with the H1 defined as a uniform distribution 
# and the likelihood defined as a t distribution.
# It also plots the Prior and Posterior (and Likelihood) and adds a pie chart.
#  
#  This is a modified version of the R script presented here:  
#  Dienes, Z., & Mclatchie, N. (2017). Four reasons to prefer Bayesian analyses
#  over significance testing. Psychonomic Bulletin & Review, 1-12. doi:
#  10.3758/s13423-017-1266-z
#
# 170601 -- Stefan Wiens
# people.su.se/~swiens/
# Thanks to Henrik Nordström, Mats Nilsson, Marco Tullio Liuzza, Anders Sand
  
# #Example
# LL = 0
# UL = 10
# meanobtained = 12
# semobtained = 5
# dfobtained = 27
# 
# BF_U(0, 10, 12, 5, 27)
# should give 6.34
#
# dfobtained = 10000
# use this to have a normal distribution as likelihood (as in Dienes online calculator)
# BF_U(0, 10, 12, 5, 10000)
# should give 7.51

{
  # Create theta (ie parameter)
  # ===========================
  theta = ((UL+LL)/2) - (2 * (UL-LL))
  tLL <- ((UL+LL)/2) - (2 * (UL-LL))
  tUL <- ((UL+LL)/2) + (2 * (UL-LL))
  incr <- (tUL - tLL) / 4000
  theta=seq(from = theta, by = incr, length = 4001)
  # The original calculator is not centered on meantheory (because the loop starts with theta + incr)
  # ie, value at position 2001 in loop does not give the meantheory
  # theta[2001]

  # Create dist_theta (ie density of prior model)
  # =============================================
  dist_theta = numeric(4001)
  dist_theta[theta>=LL & theta<=UL] = 1

  # alternative computation with normalized vectors
  dist_theta_alt = dist_theta/sum(dist_theta)
  
  # Create likelihood
  # For each theta, compute how well it predicts the obtained mean, 
  # given the obtained SEM and the obtained dfs.
  # Note that the distribution is symmetric, it does not matter if one computes
  # meanobtained-theta or theta-meanobtained
  likelihood <- dt((meanobtained-theta)/semobtained, df = dfobtained)
  # alternative computation with normalized vectors
  likelihood_alt = likelihood/sum(likelihood)

  # Multiply prior with likelihood
  # this gives the unstandardized posterior
  height <- dist_theta * likelihood
  area <- sum(height * incr)
  # area <- sum(dist_height * incr * likelihood)
  normarea <- sum(dist_theta * incr)

  # alternative computation with normalized vectors
  height_alt = dist_theta_alt * likelihood_alt
  height_alt = height_alt/sum(height_alt)

  LikelihoodTheory <- area/normarea
  LikelihoodNull <- dt(meanobtained/semobtained, df = dfobtained)
  BayesFactor <- LikelihoodTheory / LikelihoodNull

  return(BayesFactor)
  # return(c(BayesFactor, LikelihoodTheory, LikelihoodNull))

}

