import pandas as pd
data=pd.read_excel("smoke_cancer.xlsx")
df=pd.DataFrame(data)

x=df[["gender","age","smoker"]]
y=df["lung cancer"]

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x,y)

new_sample = pd.DataFrame([[0, 45, 1]], columns=["gender", "age", "smoker"])
ypred = model.predict(new_sample)

print("logistic:",ypred)

from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x,y)
print("KNeighborsClassifier:",ypred)


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x,y)

new_sample = pd.DataFrame([[0, 45, 1]], columns=["gender", "age", "smoker"])
ypred = model.predict(new_sample)


print("DecisionTreeClassifier:",ypred)
