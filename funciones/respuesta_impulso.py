import numpy as np
import soundfile as sf
from scipy.fftpack import fft, ifft
import matplotlib.pyplot as plt
from scipy.signal import chirp, fftconvolve
import scipy.signal as signal
import scipy as sc
import os




def sinesweep(duration, start_freq, end_freq, fs=44100):
    """
    Generates a sinewave with a logarithmic increase of its frequency over time.

    Parameters
    ----------
    duration : float
        Duration of the sinesweep signal, in seconds.
    start_freq : float
        Initial frequency of the sinewave, in Hz.
    end_freq : float
        Final frequency of the sinewave, in Hz.
    fs : int, optional
        Sample rate in Hz. The default is 44100.

    Returns
    -------
    time : numpy.ndarray
        Time vector of the sinesweep signal.
    signal : numpy.ndarray
        Amplitude vector of the sinesweep signal.
    """

    # Generate time vector
    time = np.linspace(0, duration, int(duration * fs), endpoint=False)

    # Generate sine sweep logarithmic signal
    signal = 0.5 * chirp(time, f0=start_freq, f1=end_freq, t1=duration, method='logarithmic', phi=-90)

    return time, signal

def inverse_sinesweep(duration, start_freq, end_freq, fs=44100):
    """
    This function generates the inverse filter (with its modulation) of a normalized logaritmic sine sweep for the given parameters

    Parameters
    ----------
        duration(int): sine-sweep's duration. Must be a positive integer 
        start_freq(int, float): frequency where the sine-sweep starts. Must be a positive number
        end_freq(int, float): frequency where the sine-sweep ends. Must be a positive number
        fs(int, float): sample frequency of the sine-sweep. Must be a positive number

    Output
    -------
        time(numpy.ndarray): time vector of the generated signal
        inverse_sinesweep(numpy.ndarray): inverse filter of the normalized logaritmic sine-sweep for the given parameters
    """

    #Original sine-sweep 
    time, sineswp = sinesweep(duration, start_freq, end_freq, fs)
    
    #Inversion and modulation
    R = np.log(end_freq / start_freq)  
    k = np.exp((time * R) / duration)
    inverse_sinesweep = np.flip(sineswp) / k
    
    #Normalization
    inverse_sinesweep = (inverse_sinesweep / np.max(np.abs(inverse_sinesweep)))*0.5
   
    return time, inverse_sinesweep

def inverse_filter(duration, start_freq, end_freq, fs=44100):
    """
    This function generates the inverse filter (with its modulation) of a normalized logaritmic sine sweep for the given parameters

    Parameters
    ----------
        duration(int): sine-sweep's duration. Must be a positive integer 
        start_freq(int, float): frequency where the sine-sweep starts. Must be a positive number
        end_freq(int, float): frequency where the sine-sweep ends. Must be a positive number
        fs(int, float): sample frequency of the sine-sweep. Must be a positive number

    Output
    -------
        time(numpy.ndarray): time vector of the generated signal
        inverse_sinesweep(numpy.ndarray): inverse filter of the normalized logaritmic sine-sweep for the given parameters
    """
    #Original sine-sweep 
    time, sineswp = sinesweep(duration, start_freq, end_freq, fs)
    
    #Inversion and modulation
    R = np.log(end_freq / start_freq)  
    k = np.exp((time * R) / duration)
    inverse_sinesweep = np.flip(sineswp) / k
    
    #Normalization
    inverse_sinesweep = inverse_sinesweep / np.max(np.abs(inverse_sinesweep))

   
    return time, inverse_sinesweep

def generar_filtro_inverso(input_file, output_file, fs=44100):
     # Ruta completa para guardar el archivo .wav
    output_path = os.path.join("Audios", output_file)

    # Cargar el archivo .wav del sine sweep
    sine_sweep, fs_sine_sweep = sf.read(input_file)

    # Verificar si el archivo de entrada es estéreo o mono
    if sine_sweep.ndim == 1:
        # Señal mono
        mono_sine_sweep = sine_sweep
    else:
        # Señal estéreo, convertir a mono promediando los canales
        mono_sine_sweep = np.mean(sine_sweep, axis=1)

    duracion = len(mono_sine_sweep) / fs_sine_sweep  # Duración del sine sweep

    t_swipe_arange = np.arange(0, duracion * fs) / fs  # Arreglo de muestreos
    R = np.log(11500 / 88)  
    K = duracion * 2 * np.pi * 88 / R
    L = duracion / R
    w = (K / L) * np.exp(t_swipe_arange / L)
    m = 88 / w

    # Calcula el filtro inverso k(t)
    k_t = m * mono_sine_sweep[::-1]  # Inversión temporal de x(t)

    # Normaliza el Filtro Inverso
    k_t /= np.max(np.abs(k_t))

    # Guarda el filtro inverso k(t) como archivo de audio .wav
    sf.write(output_path, k_t.astype(np.float32), fs)

    return k_t

def respuesta_al_impulso(sine_sweep, filtro_inverso, salida_wav=None, fs=44100):
    """
    Calcula la respuesta al impulso de una convolución entre un sine sweep y su filtro inverso.

    Parameters:
    ----------
    sine_sweep : list or np.ndarray
        Lista o array de NumPy de la señal del sine sweep.
    filtro_inverso : list or np.ndarray
        Lista o array de NumPy de la señal del filtro inverso.
    salida_wav : str, optional
        Nombre del archivo de salida .wav para guardar la respuesta al impulso. Si es None, no se guarda el archivo.
    fs : int
        Frecuencia de muestreo. Necesaria para guardar el archivo de salida.

    Returns:
    -------
    respuesta_impulso : np.ndarray
        La respuesta al impulso resultante de la convolución.
    """

    # Convertir las listas a arrays de NumPy si es necesario
    if isinstance(sine_sweep, list):
        sine_sweep = np.array(sine_sweep)
    if isinstance(filtro_inverso, list):
        filtro_inverso = np.array(filtro_inverso)

    # Asegurarse de que ambas señales tengan la misma forma
    if sine_sweep.ndim == 1:
        sine_sweep = np.expand_dims(sine_sweep, axis=-1)
    if filtro_inverso.ndim == 1:
        filtro_inverso = np.expand_dims(filtro_inverso, axis=-1)

    if sine_sweep.shape[1] != filtro_inverso.shape[1]:
        if sine_sweep.shape[1] == 1:
            sine_sweep = np.tile(sine_sweep, (1, filtro_inverso.shape[1]))
        elif filtro_inverso.shape[1] == 1:
            filtro_inverso = np.tile(filtro_inverso, (1, sine_sweep.shape[1]))
        else:
            raise ValueError("Las señales deben tener el mismo número de canales.")

    # Calcular la transformada de Fourier de ambas señales
    fft_sine_sweep = fft(sine_sweep, axis=0)
    fft_filtro_inverso = fft(filtro_inverso, axis=0)

    # Multiplicar las transformadas en el dominio de la frecuencia
    respuesta_frecuencial = fft_sine_sweep * fft_filtro_inverso

    # Aplicar la antitransformada para obtener la respuesta al impulso en el dominio del tiempo
    respuesta_impulso = ifft(respuesta_frecuencial, axis=0).real

    # Normalizar la respuesta al impulso
    respuesta_impulso /= np.max(np.abs(respuesta_impulso))

    # Guardar la respuesta al impulso como archivo de audio .wav si se especifica salida_wav
    if salida_wav is not None:
        os.makedirs("Audios", exist_ok=True)
        sf.write(os.path.join("Audios", salida_wav), respuesta_impulso, fs)

    return respuesta_impulso

def convolve(signal1, signal2):
    """
    Convolutions two signals in the time domain and returns its normalized amplitude array.

    Parameters
    ----------
    signal1 : numpy.ndarray
        amplitude array of first signaL.
    signal2 : numpy.ndarray
        amplitude array of second signal.

    Returns
    -------
    conv : numpy.ndarray
        normalizedn amplitude array of the convolution.

    """
    #Convolve
    ir = fftconvolve(signal1, signal2, mode="full")
    
    #Normalize
    ir = ir/np.max(ir)
    
    return ir

def encontrar_desfase(audio1, audio2):
    """
    Find the time offset (delay) between two audio signals using cross-correlation.

    Parameters:
    ----------
    audio1 : np.ndarray
        NumPy array containing the first audio signal.
    audio2 : np.ndarray
        NumPy array containing the second audio signal.

    Returns:
    -------
    desfase : int
        Number of samples that audio2 is offset relative to audio1.

    Description:
    ------------
    This function computes the cross-correlation between two audio signals to find the offset (delay) in samples. The cross-correlation is calculated using `signal.correlate` from the SciPy library, and the offset is determined by finding the index of the maximum value in the cross-correlation array. The function prints the offset in samples and returns this value.
    """

    # Correlación cruzada entre los dos audios
    correlacion = signal.correlate(audio2, audio1)
    desfase = np.argmax(correlacion) - len(audio1) + 1

    print(f"Existe un desfasaje entre ambas señales de {desfase} muestras")

    return desfase

def alinear_audio(audio, desfase):
    """
    Align an audio signal by adjusting for a time offset.

    Parameters:
    ----------
    audio : np.ndarray
        NumPy array containing the audio signal to be aligned.
    desfase : int
        Number of samples to adjust the audio signal by. Positive values indicate trimming from the start, while negative values indicate padding the start with zeros.

    Returns:
    -------
    audio_alineado : np.ndarray
        NumPy array containing the aligned audio signal.

    Description:
    ------------
    This function aligns an audio signal by either trimming or padding based on the provided time offset (desfase). If the offset is positive, it trims the beginning of the audio signal by the specified number of samples. If the offset is negative, it pads the beginning of the audio signal with zeros for the specified number of samples. The adjusted audio signal is returned as a NumPy array.
    """

    if desfase > 0:
        # Recortar el inicio del audio
        audio_alineado = audio[desfase:]
    else:
        # Añadir ceros al inicio del audio
        audio_alineado = np.pad(audio, (abs(desfase), 0), 'constant')
    return audio_alineado

def recortar_audio(input_wav, output_wav, duration_seconds):
    """
    Trim an audio file to a specified duration and save the trimmed audio.

    Parameters:
    ----------
    input_wav : str
        File name of the input .wav file to be trimmed.
    output_wav : str
        File name of the output .wav file to save the trimmed audio.
    duration_seconds : float
        Desired duration in seconds for the trimmed audio.

    Returns:
    -------
    audio_recortado : np.ndarray
        NumPy array containing the trimmed audio signal.

    Description:
    ------------
    This function reads an audio file from the specified input .wav file, trims it to the desired duration in seconds, and saves the trimmed audio to the specified output .wav file. The function calculates the number of samples corresponding to the desired duration based on the sampling frequency of the input audio. The trimmed audio signal is then saved and returned as a NumPy array.
    """

    # Leer el archivo de audio
    audio, fs = sf.read(f"Audios/{input_wav}")

    # Calcular el número de muestras correspondientes a la duración deseada
    num_samples = int(duration_seconds * fs)

    # Recortar el audio a la duración deseada
    audio_recortado = audio[:num_samples]

    # Guardar el audio recortado
    sf.write(f"Audios/{output_wav}", audio_recortado, fs)

    return audio_recortado

def recortar_audios(audio1, audio2, save_file_1=None, save_file_2=None, sr=44100):
    """
    Trim two audio signals to the same duration and optionally save them as separate .wav files.

    Parameters:
    ----------
    audio1 : np.ndarray
        NumPy array containing the first audio signal to be trimmed.
    audio2 : np.ndarray
        NumPy array containing the second audio signal to be trimmed.
    save_file_1 : str, optional
        File name of the output .wav file to save the trimmed first audio. If None, the file is not saved.
    save_file_2 : str, optional
        File name of the output .wav file to save the trimmed second audio. If None, the file is not saved.
    sr : int
        Sampling rate of the audio signals.

    Returns:
    -------
    audio1_recortado : np.ndarray
        NumPy array containing the trimmed first audio signal.
    audio2_recortado : np.ndarray
        NumPy array containing the trimmed second audio signal.

    Description:
    ------------
    This function ensures that both input audio signals have the same duration by trimming them to the length of the shorter signal. The trimmed audio signals are optionally saved as separate .wav files specified by 'save_file_1' and 'save_file_2'. The function returns the trimmed audio signals as NumPy arrays.
    """

    # Determine the length of the shorter audio signal
    min_len = min(len(audio1), len(audio2))

    # Trim both audio signals to the length of the shorter signal
    audio1_recortado = audio1[:min_len]
    audio2_recortado = audio2[:min_len]

    # Ensure the 'Audios' directory exists
    os.makedirs('Audios', exist_ok=True)

    # Save the trimmed audio signals to .wav files if file names are provided
    if save_file_1 is not None:
        sf.write(os.path.join('Audios', save_file_1), audio1_recortado, sr)
    if save_file_2 is not None:
        sf.write(os.path.join('Audios', save_file_2), audio2_recortado, sr)

    return audio1_recortado, audio2_recortado

def calcular_T60(respuesta_impulso_db, tasa_muestreo):
    """
    Calculate the T60 (reverberation time) of an impulse response in dB.
    
    Parameters:
    - respuesta_impulso_db: np.array, the impulse response in dB
    - tasa_muestreo: float, sampling rate in Hz

    Returns:
    - T60: float, reverberation time in seconds
    """
    # Encontrar el pico máximo y su índice
    pico_max = np.max(respuesta_impulso_db)
    indice_pico = np.argmax(respuesta_impulso_db)
    
    # Nivel de referencia (60 dB por debajo del pico)
    nivel_referencia = pico_max - 60
    
    # Buscar el punto donde la señal decae 60 dB desde el pico
    indice_T60 = np.where(respuesta_impulso_db[indice_pico:] <= nivel_referencia)[0]
    
    if len(indice_T60) == 0:
        print("La señal no decae 60 dB dentro de la duración proporcionada.")
        return None
    
    # El primer índice donde se alcanza el nivel de referencia
    indice_T60 = indice_T60[0] + indice_pico
    
    # Convertir índice a tiempo
    T60 = indice_T60 / tasa_muestreo
    
    print(f"El T60 de la señal es: {T60}")

    return T60

def correlation_lags(in1_len, in2_len, mode='full'):
    r"""
    Calculates the lag / displacement indices array for 1D cross-correlation.
    Parameters
    ----------
    in1_size : int
        First input size.
    in2_size : int
        Second input size.
    mode : str {'full', 'valid', 'same'}, optional
        A string indicating the size of the output.
        See the documentation `correlate` for more information.
    See Also
    --------
    correlate : Compute the N-dimensional cross-correlation.
    Returns
    -------
    lags : array
        Returns an array containing cross-correlation lag/displacement indices.
        Indices can be indexed with the np.argmax of the correlation to return
        the lag/displacement.
    Notes
    -----
    Cross-correlation for continuous functions :math:`f` and :math:`g` is
    defined as:
    .. math ::
        \left ( f\star g \right )\left ( \tau \right )
        \triangleq \int_{t_0}^{t_0 +T}
        \overline{f\left ( t \right )}g\left ( t+\tau \right )dt
    Where :math:`\tau` is defined as the displacement, also known as the lag.
    Cross correlation for discrete functions :math:`f` and :math:`g` is
    defined as:
    .. math ::
        \left ( f\star g \right )\left [ n \right ]
        \triangleq \sum_{-\infty}^{\infty}
        \overline{f\left [ m \right ]}g\left [ m+n \right ]
    Where :math:`n` is the lag.
    Examples
    --------
    Cross-correlation of a signal with its time-delayed self.
    >>> from scipy import signal
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> x = rng.standard_normal(1000)
    >>> y = np.concatenate([rng.standard_normal(100), x])
    >>> correlation = signal.correlate(x, y, mode="full")
    >>> lags = signal.correlation_lags(x.size, y.size, mode="full")
    >>> lag = lags[np.argmax(correlation)]
    """

    # calculate lag ranges in different modes of operation
    if mode == "full":
        # the output is the full discrete linear convolution
        # of the inputs. (Default)
        lags = np.arange(-in2_len + 1, in1_len)
    elif mode == "same":
        # the output is the same size as `in1`, centered
        # with respect to the 'full' output.
        # calculate the full output
        lags = np.arange(-in2_len + 1, in1_len)
        # determine the midpoint in the full output
        mid = lags.size // 2
        # determine lag_bound to be used with respect
        # to the midpoint
        lag_bound = in1_len // 2
        # calculate lag ranges for even and odd scenarios
        if in1_len % 2 == 0:
            lags = lags[(mid-lag_bound):(mid+lag_bound)]
        else:
            lags = lags[(mid-lag_bound):(mid+lag_bound)+1]
    elif mode == "valid":
        # the output consists only of those elements that do not
        # rely on the zero-padding. In 'valid' mode, either `in1` or `in2`
        # must be at least as large as the other in every dimension.

        # the lag_bound will be either negative or positive
        # this let's us infer how to present the lag range
        lag_bound = in1_len - in2_len
        if lag_bound >= 0:
            lags = np.arange(lag_bound + 1)
        else:
            lags = np.arange(lag_bound, 1)
    return lags
