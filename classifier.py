import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
from azureml.core import Workspace, Dataset

subscription_id = '17a98037-fffc-476a-8f95-cae56723b69e'
resource_group = 'MLOps_Weekday'
workspace_name = 'Intellipaat_MLOps'

workspace = Workspace(subscription_id, resource_group, workspace_name)

dataset = Dataset.get_by_name(workspace, name='Iris')
df = dataset.to_pandas_dataframe()
df.head()


feature= ['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
target = 'Species'

X_train, X_test, y_train, y_test = train_test_split(df[feature], df[target], test_size=0.1, shuffle=True)

clf = DecisionTreeClassifier(criterion='entropy')

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f"The accuracy of the model is {accuracy_score(y_test, y_pred)*100}")