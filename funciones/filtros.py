import numpy as np
import scipy as sc

#Creamos listas con las frecuencias centrales de bandas de octava y tercio de octava normalizadas
nominal_octave_fms = [125, 250, 500, 1000, 2000, 4000, 8000]
exact_octave_fms= [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
nominal_third_fms = [80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000]
exact_third_fms = [78.74506, 99.21256, 125.00000, 157.49013, 198.42513, 250.00000, 314.98026, 396.85026, 500.00000, 629.96052, 793.70052, 1000.00000, 1259.92104, 1587.40105, 2000.00000, 2519.84209, 3174.80210, 4000.00000, 5039.68419, 6349.60420, 8000.00000, 10079.36839]


def filter_creator(f0, sr, b=1, order=6):
    """
    Create an octave filter centered on frequency f0.
    Input:
        - f0: float. Center frequency of the octave filter.
        - fs: float. Sampling frequency of the input signal.
        - b = int. indicates the width of the band. Default is 1, creates octave band. 
        - order: int. Butterworth filter order. By default it is 4.
    Output:
        - sos: array. Second order sections of the filter.
    """
    f1 = 2**(-1/(2*b))*f0
    f2 = 2**(1/(2*b))*f0

    fc1 = f1/(sr*0.5)
    fc2 = f2/(sr*0.5)

    # Verificar que las frecuencias normalizadas están en el rango correcto
    if not (0 < fc1 < 1) or not (0 < fc2 < 1):
        raise ValueError("Las frecuencias críticas deben estar en el rango de 0 a 1")
    
    sos = sc.signal.butter(order, [fc1, fc2], btype='bandpass', output="sos")

    return sos

def freq_response_filters(filters, sr):
    freqs_responses = []
    for i in range(len(filters)):
        sos = filters[i]
        w, H = sc.signal.sosfreqz(sos, worN=16384)
        f = (w * sr) / (2 * np.pi)
        eps = np.finfo(float).eps
        H_mag = 20 * np.log10(abs(H) + eps)
        freq_response_plot = {"x": f, "data": H_mag, "xscale": "log", "legend": f"Filter {i+1}"}
        freqs_responses.append(freq_response_plot)
    return freqs_responses


        
def filter_applier(sos_list, signal):
    """
    Applies multiple digital filters specified by SOS coefficients to a signal.
    
    Parameters:
    -----------
    sos_list : list of array_like
        List of SOS filter coefficients.
    
    signal : array_like
        Input signal to filter. Should be a 1-D or 2-D array.

    Returns:
    --------
    filtered_signals : list of ndarray
        List of filtered signals after applying each SOS digital filter.
    """
    filtered_signals = []
    
    for sos in sos_list:
        filtered_signal = sc.signal.sosfilt(sos, signal)
        filtered_signals.append(filtered_signal)
    
    return filtered_signals
    
