from collections import defaultdict
from sklearn import tree
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

class FusionLearner(object):

    def __init__(self, classifier_families):
        self.classifier_families = classifier_families
        self.result_grid = defaultdict(list)

    @staticmethod
    def candidate_families():
        """
        List of candidate family classifiers with parameters for grid search
        [name, classifier object, parameters].
        """
        candidates = []
        svm_tuned_parameters = [{'kernel': ['linear'],  'C': [0.1, 1, 10, 100, 1000]},
                                {'kernel': ['rbf'],     'C': [0.1, 1, 10, 100, 1000]}]
        candidates.append(["SVM", SVC(C=1, class_weight='balanced'), svm_tuned_parameters])
        # dt_tuned_parameters = [{'max_features': ['auto', 'sqrt', 'log2']}]
        # candidates.append(["DecisionTree", tree.DecisionTreeClassifier(class_weight='balanced'), dt_tuned_parameters])
        return candidates

    def best_model(self, X, y):
        """
        Returns the best model from a set of model families given  training data
        using crosvalidation
        """
        self.best_quality = 0.0
        self.best_classifier = None
        self.best_params = {}
        classifiers = []
        for params, model, parameters in self.classifier_families:
            clf = GridSearchCV(model, parameters, cv=3, scoring="f1_macro", n_jobs=4, refit=True)
            clf.fit(X, y)
            best_estimator = clf.best_estimator_
            classifiers.append([clf.best_params_, clf.best_score_, best_estimator])

            print("Best parameters set found on development set:")
            print()
            print(clf.best_params_)
            print()
            print("Grid scores on development set:")
            print()
            means = clf.cv_results_['mean_test_score']
            stds = clf.cv_results_['std_test_score']
            for mean, std, params in zip(means, stds, clf.cv_results_['params']):
                self.result_grid[type(best_estimator).__name__].append((mean, std, params))
                print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
            print()

        for params, quality, classifier in classifiers:
            if (quality > self.best_quality):
                self.best_quality = quality
                self.best_params = params
                self.best_classifier = classifier

        print ("Choosen classifier %r with parameters %r" %
                (self.best_classifier, self.best_params))

        return self.best_classifier
