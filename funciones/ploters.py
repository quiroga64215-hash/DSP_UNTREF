
import numpy as np
import matplotlib.pyplot as plt 
import scipy as sc
from funciones import conversores as conv



def filters_response_ploter(signals, bw, labels=("Frecuencia [Hz]", "Amplitud"), title="Respuesta en frecuencia de los filtros"):
    """
    Parameters
    ----------
    signals : list of dict
        Lista de señales a plotear. Cada elemento de la lista debe ser un diccionario con al menos las claves 'data' (datos de la señal) y opcionalmente 'x' (frecuencias) y 'legend' (leyenda).
    bw : Int.
        indica el ancho de banda. ingrese 1 para banda de octava o 3 para tercio de octava.
    labels : tuple, optional
        Etiquetas para los ejes x e y del gráfico. Por defecto, ("Frecuencia [Hz]", "Amplitud").
    title : str, optional
        Título del gráfico. Por defecto, "Comparación de señales".
    """
    
    if bw == 1 : 
        xticks = [125, 250, 500, 1000, 2000, 4000, 8000]
    else: 
        xticks = [80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000, 10000]
        
    yticks = [-9, -6, -3, 0]
    
    plt.figure(figsize=(10, 5))
    for signal in signals:
        plt.plot(signal["x"], signal["data"], alpha=0.7)
        plt.xscale(signal.get("xscale", "linear"))

    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title(title)
    plt.xlim(80, 15000)
    plt.ylim(-7, 1)

    plt.xticks(xticks, labels=[str(tick) for tick in xticks], rotation=90)
    plt.yticks(yticks, labels=[str(tick) for tick in yticks])
    
    plt.grid(True)
    plt.show()
    
    
def normated_filter_limits(f0, sos, sr, bw):
    """
    Plots the magnitude (in dB) of a filter in frequency respect the attenuation limits.
    Inputs:
        f0: int type object. central frequency of filter
        sos: array type object. Second order sections of the filter
        sr: int type object. sample rate
        bw: int type object. Bandwidth of filter. Two possible values:
            - 1 for octave band
            - 3 for third octave band
    """

    G = 2
    f_lims = np.array([G**(-3), G**(-2), G**(-1), G**(-1/2), G**(-3/8), G**(-1/4), G**(-1/8), 1, G**(1/8), G**(1/4), G**(3/8), G**(1/2), G, G**2, G**3])
    lim_inf = [-200.0, -180.0, -80.0, -5.0, -1.3, -0.6, -0.4, -0.3, -0.4, -0.6, -1.3, -5.0, -80.0, -180.0, -200.0]
    lim_sup = [-61.0, -42.0, -17.5, -2.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, -2.0, -17.5, -42.0, -61.0]

    if bw == 1:
        x_ticks = [G**(-3), G**(-2), G**(-1), G**(-1/2), 1, G**(1/2), G, G**2, G**3]
        xtick_labels = [r'$G^{-3}$', r'$G^{-2}$', r'$G^{-1}$', r'$G^{-\frac{1}{2}}$', r'$G^{0}$', r'$G^{\frac{1}{2}}$', r'$G$', r'$G^2$', r'$G^3$']

        xlimits = (0.1, 10)
        minor_ticks = False

    elif bw == 3:
        f_lims = f_lims**(1/3)
        x_ticks = np.array([G**(-3), G**(-2), G**(-1), G**(-1/2), 1, G**(1/2), G, G**2, G**3])
        x_ticks = list(x_ticks**(1/3))

        xtick_labels = [r'$G^{-1}$', r'$G^{-\frac{2}{3}}$', r'$G^{-\frac{1}{3}}$', r'$G^{-\frac{1}{6}}$', r'$G^{0}$', r'$G^{\frac{1}{6}}$', r'$G^{\frac{1}{3}}$', r'$G^{\frac{2}{3}}$', r'$G^{1}$']

        xlimits=(0.5, 2)
        minor_ticks = True
    else:
        raise ValueError('No valid bw input. Values must be 1 for octave or 3 for third-octave band')
    
    w, H = sc.signal.sosfreqz(sos, worN=16384)
    f = (w*sr)/(2*np.pi*f0)

    eps = np.finfo(float).eps

    H_mag = 20 * np.log10(abs(H) + eps)
    plt.figure(figsize=(10, 5))
    if bw == 1:
        plt.semilogx(f, H_mag, label="Filtro de banda de octava", color="dodgerblue", alpha=0.7) 
    else:
        plt.semilogx(f, H_mag, label="Filtro de tercio de octava", color="dodgerblue", alpha=0.7)
    plt.semilogx(list(f_lims), lim_sup, label="Lim. sup. de atenuación", linestyle='dashed', color="peru") 
    plt.semilogx(list(f_lims), lim_inf, label="Lim. inf. de atenuación", linestyle='dashed', color="sandybrown")
    plt.title("Comprobación norma UNE-EN 61260 filtro Clase 1")
    plt.xticks(x_ticks, xtick_labels, minor=minor_ticks)
    plt.xlim(xlimits)
    plt.ylim(-60, 4)
    plt.legend()
    plt.grid()

    plt.xlabel("Frecuencia Normalizada")
    plt.ylabel("Nivel de Atenuación [dB]")

    plt.show()
    

def subplot_spl_curves(signals_pascals, sampling_rates, titles=None, suptitle="Nivel de presión Sonora", xlim=None, ylim=None):
    """
    Plots the Sound Pressure Level (SPL) curves for multiple signals in a grid layout with 2 columns.

    Args:
        signals_pascals (list of numpy.ndarray): List containing arrays of sound pressure level (SPL) data for each signal.
        sampling_rates (list of float): List of sampling rates corresponding to each signal.
        titles (list of str, optional): List of titles for each subplot. Default is None.
        suptitle (str, optional): Super title for the entire plot. Default is "Nivel de presión Sonora".
        xlim (tuple of float, optional): Limits for the x-axis. Default is None.
        ylim (tuple of float, optional): Limits for the y-axis. Default is None.

    Raises:
        ValueError: If lengths of signals_pascals and sampling_rates do not match.
        ValueError: If length of titles does not match the number of signals.
    """
    num_signals = len(signals_pascals)
    if len(sampling_rates) != num_signals:
        raise ValueError("Lengths of signals_pascals and sampling_rates must be the same.")

    if titles is None:
        titles = [f"Signal {i+1}" for i in range(num_signals)]
    elif len(titles) != num_signals:
        raise ValueError("Length of titles must match the number of signals.")

    # Determine the number of rows needed for the subplot
    num_rows = (num_signals + 1) // 2

    # Create figure and axes based on the number of signals
    fig, axs = plt.subplots(num_rows, 2, figsize=(12, num_rows * 4))

    # Ensure axs is always a 2D array even if there's only one row
    if num_rows == 1:
        axs = np.expand_dims(axs, axis=0)

    for idx, (signal, rate, title) in enumerate(zip(signals_pascals, sampling_rates, titles)):
        row = idx // 2
        col = idx % 2
        time_axis = np.arange(len(signal)) / rate
        
        # Convert signal in Pascals to SPL in dB
        spl_signal = conv.db_spl(signal)
        
        axs[row, col].plot(time_axis, spl_signal)
        axs[row, col].set_xlabel("Tiempo [s]", fontsize=12)
        axs[row, col].set_ylabel("NPS [dB]", fontsize=12)
        if xlim is not None:
            axs[row, col].set_xlim(xlim)
        if ylim is not None:
            axs[row, col].set_ylim(ylim)
        else:
            axs[row, col].set_ylim(50, 110)
        axs[row, col].set_title(title, fontsize=14)
        axs[row, col].grid(True)
        
        # Calculate LEQ and plot as red dashed line
        leq_value = conv.leq(signal)
        axs[row, col].axhline(leq_value, color='r', linestyle='--', label=f'LEQ: {leq_value:.2f} dB', zorder=5)
        axs[row, col].legend(fontsize=12)

    # Turn off any unused subplots if the total number of plots is odd
    if num_signals % 2 != 0:
        axs[-1, -1].axis('off')

    fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust subplot layout
    plt.show()
    
    
def plot_leq_band_comparison(signal_leqs1, signal_leqs2, bw, xlim=None, ylim=None, labels=None, title=None, ax=None):
    """
    Plot LEQ values for each band in a bar chart with logarithmic x-axis scale for two signals.

    Parameters
    ----------
    signal_leqs1 : list of float
        LEQ values for the first signal for each band.
        
    signal_leqs2 : list of float
        LEQ values for the second signal for each band.
        
    bw : int
        Bandwidth factor. If bw=1, the bands are octaves. If bw=3, the bands are thirds.
    
    xlim : tuple, optional
        Tuple (xmin, xmax) specifying the limits of the x-axis.
        
    ylim : tuple, optional
        Tuple (ymin, ymax) specifying the limits of the y-axis.
        
    labels : list of str, optional
        List of labels for the two signals. Default is ["Signal 1", "Signal 2"].
        
    suptitle : str, optional
        Suptitle for the plot.
        
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, a new figure and axes will be created.
    
    Returns
    -------
    None
    """
    if labels is None:
        labels = ['Signal 1', 'Signal 2']

    if bw == 1:
        band_centers = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
        band_labels = ['125', '250', '500', '1k', '2k', '4k', '8k']
        bar_widths = band_centers / 2
    elif bw == 3:
        band_centers = np.array([100, 125, 160, 200, 250, 315, 400, 500, 
                                 630, 800, 1000, 1250, 1600, 2000, 2500, 
                                 3150, 4000, 5000, 6300, 8000, 10000])
        band_labels = ['100', '125', '160', '200', '250', '315', '400', '500', 
                       '630', '800', '1k', '1.25k', '1.6k', '2k', '2.5k', '3.15k', 
                       '4k', '5k', '6.3k', '8k', '10k']
        bar_widths = band_centers / 6
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    bars1 = ax.bar(band_centers, signal_leqs1, width=bar_widths, align='center', alpha=0.8, color='green', label=labels[0])
    bars2 = ax.bar(band_centers, signal_leqs2, width=bar_widths, align='center', alpha=0.8, color='orange', label=labels[1])

    # Configurar escala logarítmica en el eje x
    ax.set_xscale('log')

    # Ajustar los ticks del eje x y las etiquetas
    ax.set_xticks(band_centers)
    ax.set_xticklabels(band_labels, rotation=90, fontsize=12)  # Tamaño de fuente de los ticks del eje x

    # Aumentar el tamaño de los números de los ejes x
    ax.tick_params(axis='x', which='major', labelsize=14)  # Tamaño de los números del eje x
    ax.tick_params(axis='y', which='major', labelsize=14)  # Tamaño de los números del eje y
    # Añadir etiquetas y título
    ax.set_xlabel('Frecuencia Central (Hz)', fontsize=14)  # Tamaño de fuente del label del eje x
    ax.set_ylabel('LEQ (dB SPL)', fontsize=14)  # Tamaño de fuente del label del eje y
    ax.set_title(title, fontsize=16)  # Tamaño de fuente del título del gráfico
    
    if title:
        ax.set_title(title, fontsize=18)

    # Ajustar límites de los ejes x e y si se especifican
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Añadir leyenda
    ax.legend(fontsize=12)  # Tamaño de fuente de la leyenda

    # Mostrar el gráfico
    ax.grid(True)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)  # Ajustar el espacio entre suptitle y subplots
    
def plot_in_time(signal, fs, xlim=None, xlabel='Tiempo [s]', ylabel='Amplitud', title='Señal en Función del Tiempo', figsize=(12, 3)):
    """
    Plot a signal over time.

    Parameters:
    ----------
    signal : np.ndarray
        NumPy array containing the signal to plot.
    fs : int
        Sampling frequency of the signal.
    xlim : tuple, optional
        A tuple specifying the x-axis limits (xmin, xmax). Default is None.
    xlabel : str, optional
        Label for the x-axis. Default is 'Tiempo [s]'.
    ylabel : str, optional
        Label for the y-axis. Default is 'Amplitud'.
    title : str, optional
        Title of the plot. Default is 'Señal en el Dominio del Tiempo'.
    figsize : tuple, optional
        Figure size in inches (width, height). Default is (10, 2).

    Description:
    ------------
    This function generates a time vector based on the length of the signal and its sampling frequency.
    It then plots the signal amplitude over time, labeling the axes and adding a title to the plot.
    The plot is displayed with a grid for better readability. The x-axis limits can be set using the `xlim` parameter.
    """
    
    # Crear el vector de tiempo
    tiempo = np.linspace(0, len(signal) / fs, num=len(signal))

    # Graficar
    plt.figure(figsize=figsize)
    plt.plot(tiempo, signal)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    
    # Configurar límites del eje x si se proporciona xlim
    if xlim is not None:
        plt.xlim(xlim)
    
    plt.show()
    
    
def plot_spectrum(audios, fs=44100, N=1, xlim=(20,20000), ylim=(-60,10), title='Frequency spectrum', labels=None):
    """
    Plots the frequency spectrum of any amount of audio signals.

    Args:
        audios (numpy.ndarray or list of numpy.ndarray): A list of arrays of audio samples (or single array of audio samples).
        fs (float): The sampling frequency in Hz.
        N (int, optional): The number of samples to use for smoothing. Defaults to 1.
        xlim (tuple, optional): The x-axis limits in Hz. Defaults to (20, 20000).
        ylim (tuple, optional): The y-axis limits in dB. Defaults to (-60, 10).
        title (str, optional): Graph title. Defaults to "Frequency spectrum".
        labels (list of str, optional): List of labels for each array in audios.

    Returns:
        None
    """
    
    plt.figure(figsize=(15,5))
    if isinstance(audios, np.ndarray) and (audios.ndim == 1):
        fft_raw = np.fft.fft(audios)
        fft = fft_raw[:len(fft_raw)//2]
        fft_mag = abs(fft) #/ len(fft)

        freqs = np.linspace(0, fs/2, len(fft))

        fft_mag_norm = fft_mag / np.max(abs(fft_mag))
        eps = np.finfo(float).eps
        fft_mag_db = 20*np.log10(fft_mag_norm + eps)

        #suavizado
        ir = np.ones(N)*1/N # respuesta al impulso de MA
        smoothed_signal = sc.signal.fftconvolve(fft_mag_db, ir, mode='same')
        
        #Plot
        plt.semilogx(freqs, smoothed_signal)
        
        
    elif isinstance(audios, list) and (len(audios) > 1):
        for i in range(len(audios)):
            fft_raw = np.fft.fft(audios[i])
            fft = fft_raw[:len(fft_raw)//2]
            fft_mag = abs(fft) #/ len(fft)

            freqs = np.linspace(0, fs/2, len(fft))

            fft_mag_norm = fft_mag / np.max(abs(fft_mag))
            eps = np.finfo(float).eps
            fft_mag_db = 20*np.log10(fft_mag_norm + eps)

            #suavizado
            ir = np.ones(N)*1/N # respuesta al impulso de MA
            smoothed_signal = sc.signal.fftconvolve(fft_mag_db, ir, mode='same')
            
            #ploteo
            
            if labels != None:
                plt.semilogx(freqs, smoothed_signal, label=labels[i])
            else:
                plt.semilogx(freqs, smoothed_signal)
            
    else:
        raise TypeError("audios variable got an unexpected type object. Must be a one-dimentional array , or a list of one-dimentional arrays")
    
    ticks = [63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000, 24000]
    plt.xticks([t for t in ticks], [f'{t}' for t in ticks])
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude [dB]')
    plt.title(title)
    if labels != None:
        plt.legend()
    plt.grid()
    plt.show()
    return


def plot_audio_comparison(audios, sr, labels=None, title='Comparación de Audios'):
    """
    Plot multiple audio signals against time for comparison.

    Parameters:
    ----------
    audios : list of np.ndarray
        List of NumPy arrays containing the audio signals to plot.
    sr : int
        Sampling rate of the audio signals.
    labels : list of str, optional
        List of labels for the audio signals. Default is None.
    title : str, optional
        Title of the plot. Default is 'Comparación de Audios'.

    Description:
    ------------
    This function plots multiple audio signals against time to visually compare their waveforms.
    The time axis is generated based on the sampling rate 'sr'. The plot includes labels, a title,
    and a legend to differentiate between the signals. The plot is displayed with grid lines for better visualization.
    """

    plt.figure(figsize=(10, 6))
    
    for i, audio in enumerate(audios):
        tiempo = np.arange(len(audio)) / sr
        if labels is not None and i < len(labels):
            plt.plot(tiempo, audio, label=labels[i], alpha=0.7)
        else:
            plt.plot(tiempo, audio, label=f'Audio {i+1}', alpha=0.7)

    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


def plot_impulse_response(signal, sr=44100, xlim=(-0.1, 1)):
    """
    Plots the impulse response signal against time.

    Parameters:
    ----------
    signal : np.ndarray
        Array containing the signal values.
    sr : int, optional
        Sampling rate in Hz. Default is 44100 Hz.
    xlim : tuple, optional
        Tuple specifying the limits of the x-axis (time axis). Default is (-0.25, 1.25).

    Returns:
    -------
    None
    """
    time = np.arange(len(signal)) / sr  # Time axis in seconds
    plt.figure(figsize=(10, 6))
    plt.plot(time, signal)
    plt.title('Impulse Response')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.ylim(-0.75, 0.75)
    plt.xlim(xlim)  # Set default xlim to (-0.25, 1)
    
    plt.show()


def plot_dB_vs_time(respuesta_impulso, fs=44100, xlim=(-0.1, 1.5), ylim=(-70, 5)):
    """
    Grafica la energía de una respuesta al impulso en dB en función del tiempo.

    Parámetros:
    ----------
    respuesta_impulso : np.ndarray
        Array que contiene la respuesta al impulso.
    fs : int, opcional
        Frecuencia de muestreo en Hz. Valor por defecto es 44100 Hz.
    xlim : tuple, opcional
        Tupla que especifica los límites del eje x (tiempo). Valor por defecto es (-0.1, 1.5).
    ylim : tuple, opcional
        Tupla que especifica los límites del eje y (energía en dB). Valor por defecto es (-70, 5).

    Retorna:
    -------
    energia_db : np.ndarray
        Array que contiene la energía en dB.
    """
    # Calcular la energía de la señal (cuadrado de la amplitud)
    energia = np.square(respuesta_impulso)

    # Normalizar la energía para que el valor máximo posible sea 1
    energia /= np.max(energia)

    # Evitar problemas con valores cero
    energia[energia == 0] = 1e-10

    # Convertir la energía a dB
    energia_db = 10 * np.log10(energia)

    # Crear el vector de tiempo
    tiempo = np.linspace(0, len(energia_db) / fs, num=len(energia_db))

    # Graficar
    plt.figure(figsize=(10, 4))
    plt.plot(tiempo, energia_db)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Energía (dB)')
    plt.title('Energía en dB vs Tiempo')
    plt.grid(True)

    # Ajustar los límites del eje x
    plt.xlim(xlim)

    # Ajustar los límites del eje y
    plt.ylim(ylim)

    plt.show()

    return energia_db