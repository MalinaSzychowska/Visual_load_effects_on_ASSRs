CI_one_sample_from_descr <- function(mean, n, sd, propconf)
# propconf = 2-tailed probability entered as 0.95 for 95%
{
  nowconf = 1 - (1 - propconf)/2
  MOE <- qt(nowconf, df = n-1) * sd / sqrt(n)
  CI = cbind(mean - MOE, mean + MOE)
  return(CI)
}