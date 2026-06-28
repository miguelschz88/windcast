# ============================================
# motor.py — Lógica de WindCast
# (consulta a Open-Meteo, interpolación a altura de buje,
#  cálculo de producción y agregación de resolución)
# ============================================
import requests
import numpy as np
import pandas as pd
import json


def obtener_pronostico_viento(lat, lon, altura_buje, dias=7, modelo="best_match"):
    """
    Obtiene el pronóstico de viento interpolado a la altura de buje.

    Filosofía: SIEMPRE solicita la máxima resolución disponible (15 min).
    La agregación a resoluciones menores (horaria, diaria) se hace después
    con agregar_resolucion(), según lo que el usuario requiera.

    Elige automáticamente los dos niveles de Open-Meteo que rodean la
    altura de buje, para interpolar (no extrapolar) en cualquier parque.

    Nota: los datos de 15 min son de alta resolución real solo en Europa
    Central y Norteamérica; en otras regiones Open-Meteo los interpola
    desde la resolución horaria.

    Parámetros:
      lat, lon     : coordenadas del parque (centroide)
      altura_buje  : altura de buje en metros (objetivo de la interpolación)
      dias         : horizonte de pronóstico
      modelo       : modelo meteorológico ("best_match" elige el mejor por ubicación)

    Devuelve: DataFrame con columnas [tiempo, viento_buje (m/s)].
    """
    # Alturas a las que Open-Meteo publica viento
    NIVELES_API = [10, 80, 120, 180]

    # Seleccionar los dos niveles que rodean la altura de buje
    if altura_buje <= NIVELES_API[0]:
        h_inf, h_sup = NIVELES_API[0], NIVELES_API[1]
    elif altura_buje >= NIVELES_API[-1]:
        h_inf, h_sup = NIVELES_API[-2], NIVELES_API[-1]
    else:
        for i in range(len(NIVELES_API) - 1):
            if NIVELES_API[i] <= altura_buje <= NIVELES_API[i + 1]:
                h_inf, h_sup = NIVELES_API[i], NIVELES_API[i + 1]
                break

    # Construir la petición pidiendo SIEMPRE la máxima resolución (15 min)
    url = "https://api.open-meteo.com/v1/forecast"
    variables = f"wind_speed_{h_inf}m,wind_speed_{h_sup}m"
    params = {
        "latitude": lat, "longitude": lon,
        "wind_speed_unit": "ms", "forecast_days": dias,
        "timezone": "auto", "models": modelo,
        "minutely_15": variables,      # siempre la resolución más fina
    }

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()
    bloque = "minutely_15"

    # Interpolación logarítmica entre los dos niveles que rodean el buje
    v_inf = np.array(data[bloque][f"wind_speed_{h_inf}m"], dtype=float)
    v_sup = np.array(data[bloque][f"wind_speed_{h_sup}m"], dtype=float)
    factor = np.log(altura_buje / h_inf) / np.log(h_sup / h_inf)
    v_buje = v_inf + (v_sup - v_inf) * factor

    return pd.DataFrame({
        "tiempo": pd.to_datetime(data[bloque]["time"]),
        "viento_buje": v_buje
    })


def cargar_curva(ruta_json):
    """
    Carga una curva de potencia tabulada desde un archivo JSON.
    Devuelve: (metadatos, velocidades, potencias)
    """
    with open(ruta_json, encoding="utf-8") as f:
        modelo = json.load(f)
    curva = np.array(modelo["curva"])
    return modelo, curva[:, 0], curva[:, 1]


def calcular_produccion(df_viento, v_curva, p_curva, turbinas_disponibles):
    """
    Convierte el pronóstico de viento en producción del parque.

    Para cada instante: viento_buje -> potencia por turbina (interpolando
    sobre la curva) -> producción del parque (× turbinas disponibles).

    Añade columnas: potencia_turbina_kw, produccion_parque_kw, produccion_parque_mw.
    """
    df = df_viento.copy()
    df["potencia_turbina_kw"] = np.interp(df["viento_buje"], v_curva, p_curva)
    df["produccion_parque_kw"] = df["potencia_turbina_kw"] * turbinas_disponibles
    df["produccion_parque_mw"] = df["produccion_parque_kw"] / 1000
    return df


def agregar_resolucion(df_prod, frecuencia):
    """
    Agrega la producción a la resolución temporal deseada.
      frecuencia: "15min", "h" (horaria) o "D" (diaria)
    Promedia viento y potencia dentro de cada intervalo.
    """
    df = df_prod.set_index("tiempo")
    cols = {"viento_buje": "mean", "potencia_turbina_kw": "mean",
            "produccion_parque_mw": "mean"}
    return df.resample(frecuencia).agg(cols).reset_index()


def aplicar_curtailment(df_prod, limite_mw):
    """
    Aplica un techo de curtailment (restricción de red) a la producción.

    El operador de red limita la potencia máxima que el parque puede
    inyectar. A diferencia de la disponibilidad (que escala la producción),
    el curtailment la recorta: la producción nunca supera 'limite_mw',
    pero por debajo de ese techo no se modifica.

    Parámetros:
      df_prod   : DataFrame con la columna 'produccion_parque_mw'
      limite_mw : techo de potencia en MW (None = sin restricción)

    Devuelve el DataFrame con la producción recortada al límite.
    """
    if limite_mw is None:
        return df_prod
    df = df_prod.copy()
    # Recortar (cap) la producción al límite, sin tocar los valores menores
    df["produccion_parque_mw"] = df["produccion_parque_mw"].clip(upper=limite_mw)
    # Recalcular las columnas en kW para mantener coherencia
    df["produccion_parque_kw"] = df["produccion_parque_mw"] * 1000
    return df


def aplicar_disponibilidad_tramos(df_prod, num_turbinas, p_curva_total_kw,
                                   disponible_defecto, tramos):
    """
    Aplica disponibilidad variable por tramos de tiempo.

    Recalcula la producción del parque según cuántas turbinas están
    disponibles en cada instante. Fuera de los tramos definidos se usa
    'disponible_defecto'.

    Parámetros:
      df_prod            : DataFrame con 'tiempo' y 'potencia_turbina_kw'
      num_turbinas       : total de turbinas del grupo
      p_curva_total_kw   : (no usado aquí; se mantiene por claridad de firma)
      disponible_defecto : nº de turbinas disponibles por defecto
      tramos             : lista de dicts {inicio, fin, valor} con valor = nº turbinas

    Devuelve el DataFrame con producción recalculada.
    """
    df = df_prod.copy()
    # Columna de disponibilidad: empieza con el valor por defecto
    disp = pd.Series(disponible_defecto, index=df.index)
    for t in tramos:
        mask = (df["tiempo"] >= t["inicio"]) & (df["tiempo"] <= t["fin"])
        disp[mask] = t["valor"]
    # Recalcular producción = potencia por turbina × turbinas disponibles
    df["produccion_parque_kw"] = df["potencia_turbina_kw"] * disp
    df["produccion_parque_mw"] = df["produccion_parque_kw"] / 1000
    return df


def aplicar_curtailment_tramos(df_prod, limite_defecto_mw, tramos):
    """
    Aplica curtailment variable por tramos de tiempo.

    Recorta la producción a un límite que puede cambiar por tramo.
    Fuera de los tramos definidos se usa 'limite_defecto_mw'
    (None = sin restricción).

    Parámetros:
      df_prod           : DataFrame con 'tiempo' y 'produccion_parque_mw'
      limite_defecto_mw : límite por defecto en MW (None = sin límite)
      tramos            : lista de dicts {inicio, fin, valor} con valor = límite MW

    Devuelve el DataFrame con la producción recortada.
    """
    df = df_prod.copy()
    # Columna de límite: empieza con el valor por defecto (puede ser None)
    import numpy as np
    limite = pd.Series(np.inf, index=df.index)
    if limite_defecto_mw is not None:
        limite[:] = limite_defecto_mw
    for t in tramos:
        mask = (df["tiempo"] >= t["inicio"]) & (df["tiempo"] <= t["fin"])
        limite[mask] = t["valor"]
    # Recortar la producción al límite de cada instante
    df["produccion_parque_mw"] = np.minimum(df["produccion_parque_mw"], limite)
    df["produccion_parque_kw"] = df["produccion_parque_mw"] * 1000
    return df


def recortar_rango(df, inicio, fin):
    """
    Recorta un DataFrame (con columna 'tiempo') al rango [inicio, fin].
    Las fechas se comparan de forma robusta ante zonas horarias.
    """
    def _naive(x):
        t = pd.Timestamp(x)
        return t.tz_localize(None) if t.tzinfo is not None else t
    df = df.copy()
    t = df["tiempo"].apply(_naive)
    ini, fn = _naive(inicio), _naive(fin)
    mask = (t >= ini) & (t <= fn)
    return df[mask].reset_index(drop=True)


def obtener_zona_horaria(lat, lon):
    """
    Calcula la zona horaria IANA (ej. 'Europe/London') desde coordenadas.
    Devuelve 'UTC' si no se puede determinar.
    """
    from timezonefinder import TimezoneFinder
    tf = TimezoneFinder()
    tz = tf.timezone_at(lat=lat, lng=lon)
    return tz if tz else "UTC"


def cargar_curva_desde_archivo(archivo, nombre_archivo):
    """
    Carga una curva de potencia desde un archivo subido (CSV o JSON).
    - CSV: dos columnas (velocidad, potencia), con o sin encabezado.
    - JSON: con clave 'curva' (lista de pares) o directamente una lista de pares.

    Devuelve una lista de pares [[velocidad, potencia], ...] ordenada por velocidad.
    Lanza ValueError con un mensaje claro si el formato no es válido.
    """
    nombre = nombre_archivo.lower()

    if nombre.endswith(".json"):
        datos = json.load(archivo)
        if isinstance(datos, dict) and "curva" in datos:
            curva = datos["curva"]
        elif isinstance(datos, list):
            curva = datos
        else:
            raise ValueError("El JSON debe contener una clave 'curva' o ser una "
                             "lista de pares [velocidad, potencia].")
    elif nombre.endswith(".csv"):
        df = pd.read_csv(archivo, header=None)
        # Si la primera fila no es numérica, asumimos que es encabezado y la saltamos
        try:
            float(df.iloc[0, 0])
        except (ValueError, TypeError):
            df = df.iloc[1:]   # la primera fila era encabezado, la descartamos
        df = df.iloc[:, :2].astype(float)
        curva = df.values.tolist()
    else:
        raise ValueError("Formato no soportado. Usa un archivo .csv o .json.")

    # Validar y normalizar
    curva = [[float(v), float(p)] for v, p in curva]
    if len(curva) < 2:
        raise ValueError("La curva debe tener al menos 2 puntos.")
    curva.sort(key=lambda par: par[0])   # ordenar por velocidad
    return curva
