import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectKBest, f_classif

#####################################################################
# Author: Gus Chadney
# Date: 11/11/15
#
# Title: Predicting the number of sub-comments for a top-level
#        comment on Reddit
#
# Description: In this challenge a section of comment data from
#              Reddit was made available for analysis in an sqllite
# DB. I chose to construct my query in order to extract the number
# of sub-comments for each top-level comment in the set.  This was
# limited to a particular subreddit ("IAmA") in order to reduce the
# query time.
# Some features were engineered with the parent comment "body" field
# (text of comment) and combined with other features from the db:
#    'score' - reddit score ('up' clicks minus 'down' clicks)
#    'gilded' - reddit gilded indicator (gold)
#    'num_questions' - number of question marks in body (engineered)
#    'body_length' - body length (engineered)
#    'urgency' - number of capital letters (engineered)
# A standard linear regression algo was then applied to fit the data
# over 3 K-folds, and then the features were scored for their
# prediction accuracy
#
#####################################################################

# Initialise Variables
subreddit = "IAmA"

print 'Querying DB...\n'
con = sqlite3.connect('./input/database.sqlite')

sql = "select parent.score,\
           parent.gilded,\
           parent.body,\
           count(distinct child.name) as num_children\
       from May2015 parent, May2015 child\
       where parent.parent_id = parent.link_id\
       and parent.subreddit = \"" + subreddit + "\"\
       and parent.name = child.parent_id\
       group by parent.name"

df = pd.read_sql_query(sql, con)
con.close()

# Creating new features:
#  num_questions:  Number of question marks in comment body
#  body_length: Number of characters in body
#  urgency:  Number of CAPITAL letters in body
print 'Adding features...\n'
df['num_questions'] = df['body'].apply(lambda x: x.count('?'))
df['body_length'] = df['body'].apply(lambda x: len(x))
df['urgency'] = df['body'].apply(lambda x: sum(1 for c in x if c.isupper()))

print 'Running linear regression...'
predictors = ['score',
              'gilded',
              'num_questions',
              'body_length',
              'urgency']

# Initialising standard linear regression algo
alg = LinearRegression()
# Splitting the data into 3 K-folds
kf = KFold(df.shape[0], n_folds=3, random_state=1)
predictions = []

# Looping over the k-folds, fitting the data on the train fold
# then predicting on the test
for train, test in kf:
    train_predictors = df[predictors].iloc[train, :]
    train_target = df['num_children'].iloc[train]

    alg.fit(train_predictors, train_target)
    test_predictions = alg.predict(df[predictors].iloc[test, :])
    predictions.append(test_predictions)

# Concatenating the predictions so we can predict accuracy (by averaging)
predictions = np.concatenate(predictions, axis=0)
predictions = predictions.round()
accuracy = sum(predictions == df['num_children']) / float(len(predictions))

print 'Accuracy: %f\n' % accuracy

# Extracting the best features so we can plot them
print 'Plotting best features...\n'
selector = SelectKBest(f_classif, k='all')
selector.fit(df[predictors], df["num_children"])
scores = selector.scores_

plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.ylabel('Score')
plt.title('Best Features')
plt.savefig("reddit_comments.png")
