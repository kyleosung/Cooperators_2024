#############################################################
## McMaster - Modeling Starter Kit
## 2024 Workshop
#############################################################

#### Load Libraries
library(data.table);library(dplyr);library(ggplot2)

#### Load Claims dataset
claims = read.csv("C://Users//CE793296//OneDrive - Co-operators//Desktop//McMaster Workshop 2024//Datasets//Claims_Years_1_to_3.csv")
mean(claims$claim_amount)

get_avg_sev = function(var,data){
  
  agg = data %>% group_by(!!sym(var)) %>% summarise(Severity = mean(claim_amount),Claim_Count = n())
  scale_factor <- round(max(agg$Claim_Count ) / max(agg$Severity), digit = 2)
  ggplot(agg) +  aes_string(var,"Severity",group = 1) + geom_point() + geom_line()+
    geom_bar(aes(y = Claim_Count  / scale_factor,color = "Claim_Count "), stat = "identity", fill = "black", alpha = 0.1)
  
}

get_avg_sev("vh_age",claims)
get_avg_sev("vh_fuel",claims)
get_avg_sev("vh_type",claims)
get_avg_sev("pol_usage",claims) 
get_avg_sev("drv_sex1",claims)
get_avg_sev("drv_age1",claims)
get_avg_sev("drv_age_lic1",claims)
get_avg_sev("pol_pay_freq",claims)
get_avg_sev("pol_duration",claims)
get_avg_sev("pol_no_claims_discount",claims)
get_avg_sev("year",claims)
get_avg_sev("pol_payd",claims)

#### Create a training set, a validation set, and a test set.
#### This example splits the data based on a random sample, other splits are possible.
#### When participants submit their scores, their models should be trained using all of the data provided.
#### When participants are assessing model fit and choosing a model, 
#### best practice is to judge your own model(s) based on data that your model has not seen.

claims$random_value = sample(1:nrow(claims),nrow(claims),replace = FALSE)/nrow(claims)
train = claims %>% filter(random_value < .6)
validation = claims %>% filter(random_value >= .6 , random_value < .8)
test = claims %>% filter(random_value >= .8)

#### Create a model.
#### Claims severity follow a gamma distribution, since the values are greater than 0, and skewed right.
glm = glm(claim_amount ~ drv_age1 + vh_age, data=train,family = Gamma )
summary(glm)

#### Assess model fit
# get prediction from model on training set.
validation$Severity_Estimate = predict(glm,newdata = validation,type = "response")

#### Create something to visualize model fit
compare_predictions = function(var,data){
  
  agg = data %>% group_by(!!sym(var)) %>% summarise(Severity = mean(claim_amount),Prediction = mean(Severity_Estimate))
  
  ggplot(agg) +  aes_string(var,"Severity",group = 1) + geom_point() + geom_line(aes(color = "Actual")) +  
    geom_point(aes_string(var,"Prediction",group = 1)) + geom_line(aes_string(var,"Prediction"))
  
}

compare_predictions("drv_age1",validation)

#### Calculate RMSE for predictions
RMSE = function(x,y){
  MSE = sum((y - x)^2)/length(x)
  return(MSE^.5)
}

# This benchmark RMSE is 2193.342.
# We must iterate from here to get a better model, either with GLM, machine learning, or something else.
RMSE(validation$claim_amount,validation$claim_prediction)




