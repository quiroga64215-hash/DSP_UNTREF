import numpy as np
from funciones import filtros as fil

def rms(signal):
    """
    Calculates the Root Mean Square (RMS) of a given audio signal.

    Parameters
    ----------
    signal (array-like): The input audio signal, which can be represented as a 1D NumPy array or list of numerical values.

    Returns
    -------
    rms_value (float): The RMS value of the input signal.
    """
    squared = np.square(signal)
    mean_squared = np.mean(squared)
    rms_value = np.sqrt(mean_squared)
    return rms_value


def calibrate(signal, calibrator_signal):
    """
    Calibra una señal dividiéndola por el RMS de una señal de calibración.

    Parameters
    ----------
    signal : array-like
        La señal de entrada que se desea calibrar.
    calibrator_signal : array-like
        La señal de calibración utilizada para calcular el RMS.
    
    Returns
    -------
    calibrated_signal : array-like
        La señal calibrada.
    """
    rms_cal = rms(calibrator_signal)  # Calcular el RMS de la señal de calibración
    calibrated_signal = (signal) / rms_cal  # calibrar la señal de entrada

    return calibrated_signal


def db_spl(signal):
    """
    Calculates the sound pressure level in decibels (dB SPL) for each sample of a given audio signal.

    Parameters
    ----------
    signal (array-like): The input audio signal, which can be represented as a 1D NumPy array or list of numerical values.
    
    Returns
    -------
    spl_db (array-like): The sound pressure level in decibels (dB SPL) for each sample of the input signal.
    """
    eps = np.finfo(float).eps  # Smallest representable positive float number
    P0 = 2e-5  # Human ear's minimum audible sound pressure level in Pascals
    
    abs_signal = np.abs(signal) + eps
    signal_spl = 20 * np.log10(abs_signal / P0)

    return signal_spl


def leq(signal_cal):
    """
    Calculate the equivalent sound pressure level (LEQ) of a signal.

    Parameters
    ----------
    signal : numpy.ndarray
        The audio signal array in Pascal values.
        
    Returns
    -------
    leq : float
        The equivalent sound pressure level of the signal in dB SPL.
    """
    P0_sq = (2.0e-5)**2 
    
    signal_sq = signal_cal**2  
    
    mean_signal_sq = np.mean(signal_sq)  

    leq = 10 * np.log10(mean_signal_sq / P0_sq)

    return leq


def leq_per_band(signal, sos_list):
    """
    Calculate the LEQ for each frequency band specified by SOS filters.
    
    Parameters
    ----------
    signal : numpy.ndarray
        The audio signal array in Pascal values.
        
    sos_list : list of array_like
        List of SOS filter coefficients.
        
    Returns
    -------
    leq_bands : list of float
        List of LEQ values for each frequency band.
    """
    # Apply the filters to the signal
    filtered_signals = fil.filter_applier(sos_list, signal)
    
    # Calculate the LEQ for each filtered signal
    leq_bands = [leq(filtered_signal) for filtered_signal in filtered_signals]
    
    return leq_bands


def leq_thirds_to_octaves(leq_third_octaves):
    """
    Combina los LEQ de tercio de octava en LEQ de octava.

    Parameters
    ----------
    leq_third_octaves : list of float
        Lista con los LEQ de tercio de octava.

    Returns
    -------
    leq_octave_bands : list of float
        Lista con los LEQ de las bandas de octava resultantes.
    """
    leq_octave_bands = []

    # Procesar cada grupo de tres elementos
    for i in range(0, len(leq_third_octaves), 3):
        if i + 2 < len(leq_third_octaves):  # Asegurar que hay tres elementos
            # Tomar los tres elementos
            leq_group = leq_third_octaves[i:i+3]
            
            # Convertir los LEQ de dB a presión sonora cuadrada
            pressures_squared = [10**(leq / 10) for leq in leq_group]
            
            # Promediar las presiones sonoras cuadradas
            mean_pressure_squared = np.mean(pressures_squared)
            
            # Convertir de vuelta a dB
            leq_octave = 10 * np.log10(mean_pressure_squared)
            
            # Añadir a la lista de LEQ de octava
            leq_octave_bands.append(leq_octave)

    return leq_octave_bands
