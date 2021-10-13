CI_one_sample_for_matrix <- function(mymatrix)
{
  #mymatrix = na.omit(mymatrix)
  tmpM = colMeans(mymatrix, na.rm = TRUE)
  tmpLL = numeric(length(tmpM))
  tmpUL = numeric(length(tmpM))
  tmpn = colSums(!is.na(mymatrix))
  for (i in 1:length(tmpM)) {
    tmpttest = t.test(mymatrix[,i])
    tmpLL[i] = tmpttest$conf.int[1]
    tmpUL[i] = tmpttest$conf.int[2]
  }
  tmp2 = cbind(tmpM, tmpLL, tmpUL, tmpn)
  return(tmp2)
}
