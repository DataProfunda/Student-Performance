# Student Performance
What features determines grades of students? <br>
What the difference between female and male students? <br>
How alcohol consumption influence grades? <br>
Can we predict sex of the students? <br>

# Analysis of students performance
Two dataset concatenated together contain 1044 rows and 33 columns.

One of interesting correlation is Dalc(Daily alcohol consumption) and G3(Last grade)
![graph](https://user-images.githubusercontent.com/69935274/110383493-a6e05b80-805c-11eb-8fab-20d990bd2a5f.png)

Most of students consume alcohol max 2 times a week.
The best students consume the least amount of alcohol. 

Another interesting correlations are grades, failures, higher and studytime
![fg](https://user-images.githubusercontent.com/69935274/110384774-65e94680-805e-11eb-8f52-8ae7269e1f82.png)

There is strong correlation between failures and grades. Students with more exams failure tend to have worse grades.
Students that want to go higher in education are more likely to study more ,have less failure and have better grades.

Mean G1 < G2 < G3. Students tend to have better grades over time.

# Other factors
Female students tend to:
-study more. 

![fs](https://user-images.githubusercontent.com/69935274/110387123-79e27780-8061-11eb-8a50-87d5aadf5d5f.png)

-drink less 
![dws](https://user-images.githubusercontent.com/69935274/110387392-d0e84c80-8061-11eb-9516-184cc5a6b73e.png)

Students that drink more study less.
![dwaas](https://user-images.githubusercontent.com/69935274/110388395-515b7d00-8063-11eb-8483-d45a3d0186fa.png)

There should be some correlation between sex and grades. 

But there is no significant correlation between sex and grades!
![dwas](https://user-images.githubusercontent.com/69935274/110387742-5d930a80-8062-11eb-9fbb-8e27ec6603e2.png)

That is for me unclear why. Despite unfavourable factors(such as drinking more, study less) men have similar grades. 
From experience I know that women tend to be more righteous. 
Maybe male students are more likely to cheat on exams. Maybe male students study less but more efficient.
More data is necessary.

# Student classifier
Does features can tell us sex of the students? <br>

I've separated solution into two files: <br>
AlcoholConsumption.py - basic preprocessing, categorical featuresmapping, scaling, define MultiClassifier <br>
MultiClassifierModule.py - contains ExtraTreeClassifier, RandomForestClassifier, VotingClassifier and useful function to perform training. <br>

For hyperparameters tuning I've used GridSearchCV.  <br>
n_repetition - number of repetition for finding best classifier. <br>

# Results
Max accuracy: 94% <br>
Accuracy calculated on test data. <br>
There are some clear differences between male and female students, but in terms of grades, they have more or less  same results. <br>




