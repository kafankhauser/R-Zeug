# Analysis

#code provided to generate edx dataset
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
library(tidyverse)
library(caret)
library(data.table)
dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)





## Modify time column and data formats in edx data set and validation data set
edx$timestamp <- as.POSIXct(edx$timestamp, origin = "1970-01-01")
edx$weekday_full_name <- weekdays(edx$timestamp)
edx$month <- format(edx$timestamp, format = "%m")
edx$year <- format(edx$timestamp, format= "%Y")
edx$month <- as.numeric(edx$month)
edx$year <- as.numeric(edx$year)
edx$weekday <- recode_factor(edx$weekday_full_name, 
                             'Monday'="1",
                             'Tuesday'="2",
                             'Wednesday'="3",
                             'Thursday'="4",
                             'Friday'="5",
                             'Saturday'="6",
                             'Sunday'="7")
edx$weekday <- as.numeric(edx$weekday)
edx$genres <- as.factor(edx$genres)
edx$userId <- as.numeric(edx$userId)

validation$timestamp <- as.POSIXct(validation$timestamp, origin = "1970-01-01")
validation$year <- format(validation$timestamp, format= "%Y")
validation$year <- as.numeric(validation$year)
validation$weekday_full_name <- weekdays(validation$timestamp)
validation$month <- format(validation$timestamp, format = "%m")
validation$month <- as.numeric(validation$month)
validation$weekday <- recode_factor(validation$weekday_full_name, 
                             'Monday'="1",
                             'Tuesday'="2",
                             'Wednesday'="3",
                             'Thursday'="4",
                             'Friday'="5",
                             'Saturday'="6",
                             'Sunday'="7")
validation$weekday <- as.numeric(validation$weekday)

## Split the data into train and test set
data <- edx %>% select(rating, weekday, month, year, genres, movieId, userId)
test_index <- createDataPartition(data$rating, times = 1, p = 0.2, list = FALSE)
train_set <- data %>% slice( - test_index)
test_set <- data %>% slice(test_index)

## Baseline RMSE
baseline_mean <- mean(train_set$rating)
baseline_RMSE <- RMSE(test_set$rating, baseline_mean)
baseline_RMSE




# User effect
userId_average <- train_set %>%
  group_by(userId) %>%
  summarize(userId_mean = mean(rating - baseline_mean))

ratings_prediction <- test_set %>%
  left_join(userId_average, by = "userId") %>%
  mutate(prediction = baseline_mean + userId_mean) %>%
  pull(prediction)

RMSE_user_effect <- RMSE(ratings_prediction, test_set$rating, na.rm=T)
RMSE_user_effect

#User and Movie effect
movieId_average <- train_set %>%
  left_join(userId_average, by = "userId") %>%
  group_by(movieId) %>%
  summarize(movieId_mean = mean(rating - baseline_mean - userId_mean))

ratings_prediction <- test_set %>%
  left_join(userId_average, by = "userId") %>%
  left_join(movieId_average, by = "movieId") %>%
  mutate(prediction = baseline_mean + userId_mean + movieId_mean) %>%
  pull(prediction)

RMSE_user_movie_effect <- RMSE(ratings_prediction, test_set$rating, na.rm = T)
RMSE_user_movie_effect

# Effect of user, movie, year
genres_average <- train_set %>%
  left_join(userId_average, by = "userId") %>%
  left_join(movieId_average, by = "movieId") %>%
  group_by(genres) %>%
  summarize(genre_mean = mean(rating - baseline_mean - userId_mean - movieId_mean))

ratings_prediction <- test_set %>%
  left_join(userId_average, by = "userId") %>%
  left_join(movieId_average, by = "movieId") %>%
  left_join(genres_average, by = "genres") %>%
  mutate(prediction = baseline_mean + userId_mean + movieId_mean + genre_mean) %>%
  pull(prediction)

RMSE_user_movie_genre_effect <- RMSE(ratings_prediction, test_set$rating, na.rm=T)
RMSE_user_movie_genre_effect

# Effect of user, movie, genre, year,
year_average <- train_set %>%
  left_join(userId_average, by = "userId") %>%
  left_join(movieId_average, by = "movieId") %>%
  left_join(genres_average, by = "genres") %>%
  group_by(year) %>%
  summarize(year_mean = mean(rating - baseline_mean - userId_mean - movieId_mean - genre_mean))

ratings_prediction <- test_set %>%
  left_join(userId_average, by = "userId") %>%
  left_join(movieId_average, by = "movieId") %>%
  left_join(genres_average, by = "genres") %>%
  left_join(years_average, by = "year") %>%
  mutate(prediction = baseline_mean + userId_mean + movieId_mean + genre_mean + year_mean) %>%
  pull(prediction)

RMSE_user_movie_genre_year_effect <- RMSE(ratings_prediction, test_set$rating, na.rm = T)
RMSE_user_movie_genre_year_effect

# effect of user, movie, year, genres, month
month_average <- train_set %>%
  left_join(userId_average, by = "userId") %>%
  left_join(movieId_average, by = "movieId") %>%
  left_join(genres_average, by= "genres") %>%
  left_join(years_average, by = "year") %>%
  group_by(month) %>%
  summarize(month_mean = mean(rating - baseline_mean - userId_mean - movieId_mean - genre_mean - year_mean))

ratings_prediction <- test_set %>%
  left_join(userId_average, by = "userId") %>%
  left_join(movieId_average, by = "movieId") %>%
  left_join(genres_average, by = "genres") %>%
  left_join(years_average, by = "year") %>%
  left_join(month_average, by = "month") %>%
  mutate(prediction = baseline_mean + userId_mean + movieId_mean + genre_mean + year_mean + month_mean) %>%
  pull(prediction)

RMSE_user_movie_genre_year_month_effect <- RMSE(ratings_prediction, test_set$rating, na.rm = T)
RMSE_user_movie_genre_year_month_effect

### effect of user, movie, genre, year, month, weekday
weekday_average <- train_set %>%
  left_join(userId_average, by = "userId") %>%
  left_join(movieId_average, by = "movieId") %>%
  left_join(genres_average, by= "genres") %>%
  left_join(years_average, by = "year") %>%
  left_join(month_average, by = "month") %>%
  group_by(weekday) %>%
  summarize(day_mean = mean(rating - baseline_mean - userId_mean - movieId_mean - genre_mean - year_mean - month_mean))

ratings_prediction <- test_set %>%
  left_join(userId_average, by = "userId") %>%
  left_join(movieId_average, by = "movieId") %>%
  left_join(genres_average, by = "genres") %>%
  left_join(years_average, by = "year") %>%
  left_join(month_average, by = "month") %>%
  left_join(weekday_average, by = "weekday") %>%
  mutate(prediction = baseline_mean + userId_mean + movieId_mean + genre_mean + year_mean + month_mean + day_mean) %>%
  pull(prediction)

RMSE_user_movie_year_genre_month_day_effect <- RMSE(ratings_prediction, test_set$rating, na.rm = T)
RMSE_user_movie_year_genre_month_day_effect


# Regularisation
lambdas <- seq(0, 10, 0.5)

rmses <- sapply(lambdas, function(l){
  userId_average <- train_set %>%
    group_by(userId) %>%
    summarize(userId_mean = sum(rating - baseline_mean)/(n()+l))
  
  movieId_average <- train_set %>%
    left_join(userId_average, by = "userId") %>%
    group_by(movieId) %>%
    summarize(movieId_mean = sum(rating - baseline_mean - userId_mean)/(n()+l))
  
  genres_average <- train_set %>%
    left_join(userId_average, by = "userId") %>%
    left_join(movieId_average, by = "movieId") %>%
    group_by(genres) %>%
    summarize(genre_mean = sum(rating - baseline_mean - userId_mean - movieId_mean)/(n()+l))
  
  years_average <- train_set %>%
    left_join(userId_average, by = "userId") %>%
    left_join(movieId_average, by = "movieId") %>%
    left_join(genres_average, by = "genres") %>%
    group_by(year) %>%
    summarize(year_mean = sum(rating - baseline_mean - userId_mean - movieId_mean - genre_mean)/(n()+l))
  
  month_average <- train_set %>%
    left_join(userId_average, by = "userId") %>%
    left_join(movieId_average, by = "movieId") %>%
    left_join(years_average, by = "year") %>%
    left_join(genres_average, by= "genres") %>%
    group_by(month) %>%
    summarize(month_mean = sum(rating - baseline_mean - userId_mean - movieId_mean - genre_mean - year_mean)/(n()+l))

  weekday_average <- train_set %>%
    left_join(userId_average, by = "userId") %>%
    left_join(movieId_average, by = "movieId") %>%
    left_join(genres_average, by= "genres") %>%
    left_join(years_average, by = "year") %>%
    left_join(month_average, by = "month") %>%
    group_by(weekday) %>%
    summarize(day_mean = sum(rating - baseline_mean - userId_mean - movieId_mean - genre_mean - year_mean - month_mean)/n()+l)
  
  ratings_prediction <- test_set %>%
    left_join(userId_average, by = "userId") %>%
    left_join(movieId_average, by = "movieId") %>%
    left_join(genres_average, by = "genres") %>%
    left_join(years_average, by = "year") %>%
    left_join(month_average, by = "month") %>%
    left_join(weekday_average, by = "weekday") %>%
    mutate(prediction = baseline_mean + userId_mean + movieId_mean + genre_mean + year_mean + month_mean + day_mean) %>%
    pull(prediction)
  return(RMSE(ratings_prediction, test_set$rating, na.rm = T))
})

lambda <- lambdas[which.min(rmses)]

qplot(lambdas, rmses)  

## Validation
###Validating the final model
userId_average <- train_set %>%
  group_by(userId) %>%
  summarize(userId_mean = sum(rating - baseline_mean)/(n()+lambda))

movieId_average <- train_set %>%
  left_join(userId_average, by = "userId") %>%
  group_by(movieId) %>%
  summarize(movieId_mean = sum(rating - baseline_mean - userId_mean)/(n()+lambda))

genres_average <- train_set %>%
  left_join(userId_average, by = "userId") %>%
  left_join(movieId_average, by = "movieId") %>%
  group_by(genres) %>%
  summarize(genre_mean = sum(rating - baseline_mean - userId_mean - movieId_mean)/(n()+lambda))

years_average <- train_set %>%
  left_join(userId_average, by = "userId") %>%
  left_join(movieId_average, by = "movieId") %>%
  left_join(genres_average, by = "genres") %>%
  group_by(year) %>%
  summarize(year_mean = sum(rating - baseline_mean - userId_mean - movieId_mean - genre_mean)/(n()+lambda))

month_average <- train_set %>%
  left_join(userId_average, by = "userId") %>%
  left_join(movieId_average, by = "movieId") %>%
  left_join(years_average, by = "year") %>%
  left_join(genres_average, by= "genres") %>%
  group_by(month) %>%
  summarize(month_mean = sum(rating - baseline_mean - userId_mean - movieId_mean - genre_mean - year_mean)/(n()+lambda))

weekday_average <- train_set %>%
  left_join(userId_average, by = "userId") %>%
  left_join(movieId_average, by = "movieId") %>%
  left_join(genres_average, by= "genres") %>%
  left_join(years_average, by = "year") %>%
  left_join(month_average, by = "month") %>%
  group_by(weekday) %>%
  summarize(day_mean = sum(rating - baseline_mean - userId_mean - movieId_mean - genre_mean - year_mean - month_mean)/(n()+lambda))

ratings_prediction_validation <- validation %>%
  left_join(userId_average, by = "userId") %>%
  left_join(movieId_average, by = "movieId") %>%
  left_join(genres_average, by = "genres") %>%
  left_join(years_average, by = "year") %>%
  left_join(month_average, by = "month") %>%
  left_join(weekday_average, by = "weekday") %>%
  mutate(prediction = baseline_mean + userId_mean + movieId_mean + genre_mean + year_mean + month_mean + day_mean) %>%
  pull(prediction)

RMSE_validation <- RMSE(ratings_prediction_validation, validation$rating, na.rm=T)

## final RMSE
RMSE_validation 


