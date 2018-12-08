# linear model that is used for classification
# or estimating discrete values
# specifically data on iris flowers

from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

iris = datasets.load_iris()

# print iris.keys()
# # types of iris plants
# print iris.target_names
# # what data represents
# print iris.feature_names
# # data
# print iris.data[:3]
# # target class
# print iris.target[:3]
# # how many data points including how many features each
# print iris.data.shape
# # target also has same number of points
# print iris.target.shape
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print y_test
print predictions

# accuracy score
print model.score(X_test, y_test)

#precision: probability of positive prediction being accurately positive
#recall: probability that the model will pick up on true positive
#f1-score: is combination of precision and recall
#support: number of samples of each class in the dataset
print metrics.classification_report(y_test, predictions)

#classes on the left, actual predictions on the top
print metrics.confusion_matrix(y_test, predictions)
