import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import joblib  # For saving and loading the model

# Load the data
df = pd.read_csv('Iris.csv')

# Encode the target variable
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Prepare features and target
x = df.drop(['Species', 'Id'], axis=1)
y = df['Species']

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

# Set up the SVC with GridSearchCV
param = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
model = GridSearchCV(estimator=SVC(), param_grid=param, cv=3, verbose=1, n_jobs=-1)
model.fit(x_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'svc_model.pkl')
joblib.dump(le, 'label_encoder.pkl')  # Save the label encoder as well

