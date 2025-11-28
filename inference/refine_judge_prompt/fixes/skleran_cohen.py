from sklearn.metrics import cohen_kappa_score

#define array of ratings for both raters
# ESSE CASO DA NAN
# rater1 = [4, 4, 4, 4, 4]
# rater2 = [4, 4, 4, 4, 4]

#  CASO ONDE DA 0
# rater1 = [5,5,5,5,5]
# rater2 = [4,4,4,4,4]

#  CASO ONDE DA 0 
# rater1 = [4,4,4,4,4]	
# rater2 = [4,5,4,5,4]	

# CASO ONDE DA 1
rater1 = [1, 1, 2, 2, 2]
rater2 = [1, 1, 2, 2, 2]

#calculate Cohen's Kappa
teste = cohen_kappa_score(rater1, rater2)
print("Cohen's Kappa I:", teste)

#                     03                          
rater1 = [2, 1, 2, 2, 1]
#        03
rater2 = [1, 1, 2, 2, 2]

teste = cohen_kappa_score(rater1, rater2)
print("Cohen's Kappa II:", teste)