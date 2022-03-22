import pandas as pd
import help_functions as hf
import attribute_fuctions as af


class AttributeRunner:

    def __init__(self, case, window_size, mini_window_size, ml_object):
        # Attributes that were given by the user
        self.case = case
        self.window_size = window_size
        self.mini_window_size = mini_window_size
        self.window = []
        self.mini_window = []
        self.ml_object = ml_object
        # header der CSV-Datei die fuer das Ergebnis erstellt wird
        self.result_col = ['X',
                           'Y',
                           'Beschreibung Anomalie',
                           'Qualitative classification',
                           'mavg_window',
                           'delta_mavg_window',
                           'mavg',
                           'std',
                           'mini_std',
                           'amplitude',
                           'delta_amplitude',
                           'slope',
                           'length',
                           'is_group',
                           'id',
                           'E[Y]',
                           'delta_E[Y]',
                           'v(Y)',
                           'type']

        # Attributes die frueh erstellt werden muessen, da sie fuer die allgemeine Funktionalitaet benoetigt werden
        self.x_steps = None
        self.decimal_places = None
        self.index = 0

        # Wird gebraucht um missind data zu finden
        self.last_x = None
        # Wird verwandt um Gruppen zu erkennen
        self.length = 0
        self.id = 1


    def runner(self, row):

        # Strings in den Daten werden so angepassst, dass Umlaute entfernt werden (sorgen fuer Probleme beim CSV-Export)
        ri = 0
        for r in row:
            if type(r) is str:
                row[ri] = hf.replace_umlauts(r)
            ri += 1

        # Bevor Anomalien behandelt werden, versichere, dass die Windows nicht groesser als die Maximalgroesse werden
        if row['Anomalie ground truth'] == 0:
            self.window += [row['y']]

        # Das mini-window ist stehts um einen kleiner, da der neue Wert unten eingefuegt wird
        while len(self.mini_window) > self.mini_window_size - 1:
            self.mini_window.pop(0)
        while len(self.window) > self.window_size:
            self.window.pop(0)

        # Wenn der erste Eintrag behandelt wird ist es noch nicht moeglich die Anomalien zu behandeln
        # Result-CSV wird angelegt
        if self.index == 0:
            # Damit bei der Imputation bekannt ist ab wann aufgefuellt werden muss
            self.last_x = row['x']
            # Erstelle den Header der CSV-Datei
            head = pd.DataFrame([self.result_col], columns=self.result_col)
            print(f'create file: result/result_{self.case}')
            head.to_csv(f'result/result_{self.case}',
                        mode='w', columns=self.result_col, header=None, index=False, float_format="%.15f")
        elif self.index == 1:
            # Welches Format hat die X-Achse (Schritte um wie viel und wie viele Nachkommastellen)?
            self.x_steps, self.decimal_places = hf.get_x_steps([self.last_x, row['x']])

        else:
            if row['x'] > round(self.last_x + self.x_steps, self.decimal_places):

                # Solange die Luecke noch besteht, fuelle diese
                while row['x'] > round(self.last_x + self.x_steps, self.decimal_places):
                    self.last_x = round(self.last_x + self.x_steps, self.decimal_places)

                    # Moving Avg als einzusetzenden Wert bei fehlenden Werten
                    if len(self.window) < self.mini_window_size:
                        mavg = af.get_mavg(self.window, len(self.window), None)
                    else:
                        mavg = af.get_mavg(self.window, self.mini_window_size, None)

                    # fuege moving average in mini-window ein
                    self.mini_window += [mavg]

                    # Erstelle die Row die in die CSV geschrieben werden soll
                    # Die erforderlichen Spalten werden ergaenzt, rest bleibt aus Kopie erhalten
                    imput_row = row.copy(deep=True)
                    imput_row['x'] = self.last_x
                    imput_row['y'] = mavg
                    imput_row['Beschreibung Anomalie'] = None
                    imput_row['Anomalie ground truth'] = 0
                    imput_row['Qualitative classification'] = 0

                    self.length += 1
                    result_row = self.get_attributes(imput_row, self.length, self.id, self.result_col, self.window, self.mini_window, 'imputated')
                    result_row.to_csv(f'result/result_{self.case}',
                                      mode='a', columns=self.result_col, header=None, index=False, float_format="%.15f")

            if row['Anomalie ground truth'] == 1:
                self.mini_window += [row['y']]

                self.length += 1
                result_row = self.get_attributes(row, self.length, self.id, self.result_col, self.window, self.mini_window, None)
                result_row.to_csv(f'result/result_{self.case}',
                                  mode='a', columns=self.result_col, header=None, index=False, float_format="%.15f")

            else:
                # Wenn es keine Anomalie ist, beginnt die laenge der naechsten Anomalie wieder bei null, ausserdem aendert sich die ID
                if self.length > 0:
                    self.id += 1

                self.length = 0

        self.last_x = row['x']
        self.index += 1

    # Bestimme alle Attribute nacheinander
    def get_attributes(self, row, length, id, col, window, mini_window, type):

        if len(window) == 0:
            window = [0]
        if len(mini_window) == 0:
            mini_window = [0]

        dummy = []
        for i in col:
            dummy += [None]

        result = pd.DataFrame([dummy], columns=col)

        result['X'] = row['x']
        result['Y'] = row['y']
        result['Beschreibung Anomalie'] = row['Beschreibung Anomalie']
        result['Qualitative classification'] = row['Qualitative classification']
        mavg_window = af.get_mavg(window, len(window), None)
        result['mavg_window'] = mavg_window
        result['delta_mavg_window'] = mavg_window - row['y']
        mavg = af.get_mavg(mini_window, len(mini_window), None)
        result['mavg'] = mavg
        std = af.get_std(window)
        result['std'] = std
        mini_std = af.get_std(mini_window)
        result['mini_std'] = mini_std
        amplitude = af.get_amplitude(window, mavg_window)
        result['amplitude'] = amplitude
        result['delta_amplitude'] = amplitude - row['y']
        slope = af.get_slope(mini_window)
        result['slope'] = slope
        result['length'] = length
        if length > 1:
            result['is_group'] = 1
        else:
            result['is_group'] = 0
        result['id'] = id
        expected_value = af.get_expected_value(mini_window)
        result['E[Y]'] = expected_value
        result['delta_E[Y]'] = expected_value - row['y']
        result['v(Y)'] = af.get_tilt(mini_window, mavg, std)
        if type is None:
            result['type'] = self.ml_object.predict_with_attributes(result)
        else:
            result['type'] = type

        return result
