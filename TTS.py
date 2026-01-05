import sounddevice as sd
import soundfile as sf
from gradio_client import Client, handle_file


def TextToSpeech(texto):
    # ¡¡API TTS!!
    DURACION = 20 
    FRECUENCIA_MUESTREO = 44100 

    # ARCHIVO_REFERENCIA = "ejemplo.wav"
    ARCHIVO_REFERENCIA = "ref_audio.wav"

    if ARCHIVO_REFERENCIA=="ejemplo.wav":
        pass
    else:
        print("Grabando audio de referencia...")
        print("Habla todo lo que puedas durante 20s.")
        audio = sd.rec(int(DURACION * FRECUENCIA_MUESTREO), samplerate=FRECUENCIA_MUESTREO, channels=1)
        sd.wait()
        sf.write(ARCHIVO_REFERENCIA, audio, FRECUENCIA_MUESTREO)
        print("Grabación finalizada.")


    client = Client("redradios/F5-TTS-Sp", "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")
    result = client.predict(
        ref_audio_orig=handle_file(ARCHIVO_REFERENCIA),
        ref_text="",
        gen_text=texto.lower() + ".",
        model="F5-TTS",
        remove_silence=False,
        cross_fade_duration=0.15,
        speed=1.0,
        api_name="/infer"
    )

    ruta_audio = result[0]
    x, FRECUENCIA_MUESTREO = sf.read(ruta_audio)

    print("Reproduciendo audio generado...")
    sd.play(x, FRECUENCIA_MUESTREO)
    sd.wait()

    salida = "audio_generado.wav"
    sf.write(salida, x, FRECUENCIA_MUESTREO)
    print(f"Audio generado guardado como {salida}")
