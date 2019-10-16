import numpy as np
import pandas as pd
import pickle
import sklearn.ensemble as ske
from sklearn import cross_validation, tree, linear_model
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

data = pd.read_csv('MalwareData.csv', sep = '|')
X = data.drop(['Name', 'md5', 'legitimate'], axis=1).values
Y = data['legitimate'].values

print('Researching important feature based on %i total features\n' % X.shape[1])

fsel = ske.ExtraTreesClassifier().fit(X, Y)
model = SelectFromModel(fsel, prefit=True)
X_new = model.transform(X)
nb_features = X_new.shape[1]
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X_new, Y, test_size=0.2)

features = []

print('%i features identifed as important :' % nb_features)

indices = np.argsort(fsel.feature_importances_)[::-1][:nb_features]

for f in range(nb_features):
    print("%d. feature %s (%f)" % (f + 1, data.columns[2+indices[f]], fsel.feature_importances_[indices[f]]))

algorithms = {
    "DecisionTree": tree.DecisionTreeClassifier(max_depth=10),
    "RandomForest": ske.RandomForestClassifier(n_estimators=50),
    "GradientBoosting": ske.GradientBoostingClassifier(n_estimators=50),
    "AdaBoost": ske.AdaBoostClassifier(n_estimators=100),
    "GNB": GaussianNB()
}

results = {}
print("\nNow testing algorithms")
for algo in algorithms:
    clf = algorithms[algo]
    clf.fit(X_train, Y_train)
    score = clf.score(X_test, Y_test)
    print("%s : %f %%" % (algo, score*100))
    results[algo] = score

winner = max(results, key = results.get)
print('\nWinner algorithm is %s with a %f %% success' % (winner, results[winner]*100))

print('Saving algorithm and feature list in classifier directory...')
joblib.dump(algorithms[winner], 'classifier/claasifier.pkl')
open('classifier/features.pkl' 'w').write(pickle.dumps(features))
print('Saved')

clf = algorithms[winner]

res = clf.predict(X_test)
mt = confusion_matrix(Y_test, res)
print("False positive rate : %f %%" % ((mt[0][1] / float(sum(mt[0])))*100))
print('False negative rate : %f %%' % ((mt[1][0] / float(sum(mt[1]))*100)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Detect malicious file from manalyzer infos')

parser.add_argument('URL', help = 'Manalyzer url')

args = parser.parse_args()

clf = joblib.load(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'classifier/classifier.pkl'
))
features = pickle.loads(open(os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'classifier/features.pkl'),
    'r').read()
)
