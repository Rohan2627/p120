from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model import LogisticRegression 

from sklearn import datasets

# ----------------------------------------------------------------------------------
wine = datasets.load_wine()

print("Features : " , wine.feature_names)

print("Labels : " , wine.target_names)


X = wine.data
y = wine.target


x_train , x_test , y_train , y_test = train_test_split(X , y , test_size=0.25)

sc = StandardScaler()

x_train = sc.fit_transform(x_train)

x_test = sc.fit_transform(x_test)


classifier = GaussianNB()

classifier.fit(x_train,y_train)



predict_data = classifier.predict(x_test)


print("Accuracy Score using Gaussian NB:", accuracy_score(y_test, predict_data))













