import numpy as np

# Enthaelt alle Methoden die fuer die Attributberechnung verwandt werden

# Berechne einen Moving Average (fuer mavg_window und mavg)
def get_mavg(window, window_size, y):
    if len(window) == 0:
        return 0
    else:
        if y is not None:
            window += [y]
        w = np.asarray(window[len(window) - window_size:])
        avg = w.mean()

        return avg

# Berechne die Standardabweichung fuer das Fenster (fuer std und mini_std)
def get_std(window):
    if len(window) == 0:
        return 0
    else:
        std = np.array(window).std()
        return std

# Bestimmte die Amplitude
def get_amplitude(window, avg):
    if len(window) == 0:
        return 0
    else:
        max_a = abs(max(window)-float(avg))
        min_a = abs(min(window)-float(avg))

        if max_a > min_a:
            return max_a
        else:
            return min_a

# Berechne die Steigung
def get_slope(window):
    if len(window) == 0:
        return 0
    else:
        return window[len(window)-1]-window[len(window)-2]

# Berechne den Erwartungswert
def get_expected_value(window):
    if len(window) == 0:
        return 0
    else:
        return window[len(window)-2] + get_slope(window[:len(window)-1])

# Berechne den Wert der Schiefefunktion
def get_tilt(window, avg, std):
    if len(window) == 0:
        return 0
    elif std == 0:
        return 0
    else:
        v_sum = 0
        for i in window:
            v_sum += ((i-avg)/std)**3
        return v_sum/len(window)

