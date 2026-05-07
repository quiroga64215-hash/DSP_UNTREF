import os
import soundfile as sf 
import pandas as pd


def read_wav_file(filename):
    """
    Reads a WAV file from the 'Audios' directory and returns the sample rate and signal data.
    
    Parameters:
    -----------
    filename : str
        Name of the WAV file to read.
    
    Returns:
    --------
    sample_rate : int
        Sample rate of the WAV file.
    
    signal : ndarray
        Signal data from the WAV file.
    """
    # Obtiene el directorio actual de trabajo
    current_directory = os.getcwd()
    
    # Construye la ruta completa al archivo en la carpeta 'Audios'
    file_path = os.path.join(current_directory, 'Audios', filename)
    
    signal, sample_rate = sf.read(file_path)
    return signal, sample_rate


def read_sonometer_excel(file_name, start_row, end_row):
    """
    Extrae valores de las columnas 1 y 2 de una planilla de Excel y los agrega a listas separadas.

    Parameters
    ----------
    file_name : str
        El nombre del archivo de Excel.
    start_row : int
        El número de la primera fila desde donde se comenzará a leer (1-indexed).
    end_row : int
        El número de la última fila hasta donde se leerá (1-indexed).

    Returns
    -------
    freqs_centrales : list
        Lista de frecuencias centrales.
    leq_bandas : list
        Lista de valores de LEQ por banda.
    """
    # Crear la ruta completa al archivo
    file_path = os.path.join("Excels Sonometro", file_name)
    
    # Leer el archivo de Excel
    df = pd.read_excel(file_path, header=None)
    
    # Ajustar los índices para pandas (0-indexed)
    start_row -= 1
    end_row -= 1
    
    # Extraer los valores de las columnas 1 y 2
    freqs_centrales = df.iloc[start_row:end_row+1, 0].tolist()
    leq_bandas = df.iloc[start_row:end_row+1, 1].tolist()
    
    return freqs_centrales, leq_bandas


def cut_signal(signal, sr_signal, t0=None, tf=None):
    """
    Corta una señal en un intervalo de tiempo específico.

    Parameters 
    ------------
    signal (array-like): La señal de entrada.
    sr_signal (int): La tasa de muestreo de la señal.
    t0 (float, opcional): El tiempo de inicio del corte en segundos. Por defecto es None.
    tf (float, opcional): El tiempo de finalización del corte en segundos. Por defecto es None.

    Returns
    -----------------
    signal_cut (array-like): La señal cortada entre t0 y tf.
    
    """

    # Si t0 es None, establecerlo al inicio de la señal
    if t0 is None:
        t0 = 0

    # Si tf es None, establecerlo al final de la señal
    if tf is None:
        tf = len(signal) / sr_signal

    # Convertir los tiempos t0 y tf a índices de muestra
    start_idx = int(t0 * sr_signal)
    end_idx = int(tf * sr_signal)

    # Cortar la señal
    signal_cut = signal[start_idx:end_idx]

    return signal_cut
