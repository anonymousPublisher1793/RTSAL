import pandas as pd
import numpy as np
import help_functions as hf
# import Decision Trees
from sklearn.tree import DecisionTreeClassifier
# import kNN
from sklearn.neighbors import KNeighborsClassifier
# import Naive Bayes
from sklearn.naive_bayes import BernoulliNB
# import MLPClassifier
from sklearn.neural_network import MLPClassifier

# Erstelle ein Objekt, dass alle ML-Modelle enthaelt und Label Predicted
class Classification:

    def __init__(self, case, algorithm, repeats, attributes):
        self.case = case
        self.algorithm = algorithm
        self.repeats = repeats
        # Liste in der die reapets-vielen ML-Modelle enthalten sein werden
        self.object_list = []
        # Liste in der alle Attribute, die verwandt werden sollen enthalten sein werden
        self.attributes = []

        # Waehle die Attribute, die verwandt werden sollen
        if attributes == 'pre_optimized_1':
            self.attributes = hf.get_pre_optimized_1()
        elif attributes == 'pre_optimized_2':
            self.attributes = hf.get_pre_optimized_2()
        elif attributes == 'manual_select':
            self.read_mas_file()
        # Die letzte moeglichkeit ist, dass alle gewaehlt werden
        else:
            self.attributes = hf.get_attributes()

    # Wird aufgerufen um den Learning-Prozess zu beginnen
    # Zuerst werden die Daten gelesen und Stichproben gezogen (stratified sampling)
    # Dann werden die repeats-vielen Modelle trainiert und in der object_list gespeichert
    def start_learning(self):
        # Nur die gewahelten Attribute werden beim Training beachtet
        include = self.attributes

        # Lese die gelabelten Daten ein, "imputated" wird geloescht, die dieses Label spaeter bei Laufzeit bestimmt
        # und nicht predicted wird
        data = pd.read_csv(f'labelled_data/{self.case}', delimiter=';', header=0)
        data = data.drop(data[data['type'] == 'imputated'].index)
        data.index = list(range(len(data['type'])))

        # Gehe sicher, dass die Daten das richtige Format haben
        for i in data:
            data[i] = hf.str_to_float(hf.replace_comma(data[i].to_numpy()))

        # Dictionary, dass angibt, wie gross die unterschiedlichen Klassen sind
        # Die Groesse der kleinsten Klasse gibt die groesse der stichprobe vor
        class_sizes = hf.get_class_sizes(data, hf.get_types(list(data['type'])))
        sample_size = min(class_sizes.values())

        # Wenn die kleinste Klasse zu klein ist wird stattdessen min-size verwandt
        # Wenn min_size verwandt werden muss, dann muessen Dumplikate in den Stichproben erlaubt sein
        min_size = 15
        allow_duplicates = False
        if sample_size < min_size:
            sample_size = min_size
            allow_duplicates = True

        # Wiederhole das Lernen und Erstellen der Stichproben so oft wie repeats es vorgibt
        print(f'start learning on: {self.case}')
        while len(self.object_list) < self.repeats:

            sampled_data = data.groupby('type', group_keys=False).apply(lambda x: x.sample(sample_size, replace=allow_duplicates))

            x = list(sampled_data[include].to_numpy(dtype=float))
            y = list(sampled_data['type'].to_numpy())

            if self.algorithm == 'knn':
                self.object_list += [KNeighborsClassifier(n_neighbors=5, weights='distance').fit(x, y)]
            elif self.algorithm == 'naive_bayes':
                self.object_list += [BernoulliNB().fit(x, y)]
            elif self.algorithm == 'mlp_classifier':
                self.object_list += [MLPClassifier(random_state=1, max_iter=300).fit(x, y)]
            # Decision Trees ist der Standard
            else:
                self.object_list += [DecisionTreeClassifier(random_state=0, class_weight='balanced').fit(x, y)]

            if len(self.object_list)%10 == 0:
                print(f'progress for {self.case}: {round((len(self.object_list)/self.repeats)*100, 2)}%')



    # An dieser Stelle wuerde ein Prozess stehen, der den optimalen ML-Algo findet
    # Decision Trees als Standard ist der Platzhalter
    def ml_optimize(self):
        return 'decision_trees'

    # Methode mit der der Zustand der Modelle bei der Implementierung getestet wurde
    def ml_status(self):
        print(f'status for learning on {self.case}')
        print(f'Attributes: {self.attributes}')
        print(f'Algortihm: {self.algorithm}')
        print(f'Objects in List: {len(self.object_list)}/{self.repeats}')

    # Wenn die Attribute manuell gewaehlt werden sollen, dann lese diese aus den .txt-Dateien aus
    def read_mas_file(self):
        try:
            # Die Attribute muessen sich in den .txt-Dateien befinden, die wie die Dateien heissen
            with open(f'manual_attribute_selection/{self.case[:len(self.case) - 4]}.txt') as file:
                file_txt = file.read()
        except IOError:
            print('ERROR: manuel_attribute_selection file does not exist')
            quit()

        file_txt = file_txt.replace('\n', '')
        file_txt = file_txt.replace(' ', '')
        file_list = file_txt.split(',')
        for fl in file_list:
            # Nur existierende Attribute, die kein Dulikat sind koennen verwandt werden
            if fl in hf.get_attributes() and fl not in self.attributes:
                self.attributes += [fl]
        if len(self.attributes) < 2:
            print('ERROR: there have to be at least two viable attributes')
            quit()

    # Erstelle die Prediction mit der Sicherheit fuer das Ergebnis in Prozent
    def predict_with_attributes(self, all_attributes):
        prediction_list = []
        class_dic = {}
        to_be_predicted = [list(all_attributes.loc[0][self.attributes]), ]

        # Wiederhole fuer alle Modelle in der Liste
        for ol in self.object_list:
            prediction = ol.predict_proba(to_be_predicted)
            j = 0
            for c in ol.classes_:
                if c not in class_dic:
                    class_dic[c] = [prediction[0][j]]
                else:
                    class_dic[c] += [prediction[0][j]]
                j += 1
            prediction_list += [prediction[0]]

        # Finde das Attribut mit der hoechsten durchschnittlichen Sicherheit
        highest_mean = 0
        highest_mean_key = 'Label'
        for key in class_dic:
            current_mean = np.asarray(class_dic[key]).mean()
            if current_mean > highest_mean:
                highest_mean = round(current_mean, 4)
                highest_mean_key = key

        print(f'in {self.case}: {highest_mean_key} ({round(highest_mean * 100, 4)}%)')
        return f'{highest_mean_key} ({highest_mean * 100}%)'
