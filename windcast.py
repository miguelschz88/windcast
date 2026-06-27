# ============================================
# windcast.py — Aplicación WindCast
# ============================================
import streamlit as st
import json
import os
import glob
import datetime as dt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
import motor   # nuestras funciones del motor

st.set_page_config(page_title="WindCast", page_icon="🌬️", layout="wide")


# --- Funciones auxiliares para manejar parques ---
def listar_parques():
    """Devuelve la lista de archivos de parques guardados."""
    return sorted(glob.glob("parques/*.json"))


def cargar_parque(ruta):
    """Carga un parque desde su archivo JSON."""
    with open(ruta, encoding="utf-8") as f:
        return json.load(f)


def validar_tramos(tabla, campo_valor, valor_min, valor_max, etiqueta_valor,
                   inicio_horizonte=None, fin_horizonte=None):
    """
    Valida una tabla de tramos. Devuelve (errores, advertencias):
      - errores: problemas que impiden generar el pronóstico (bloquean)
      - advertencias: avisos que no bloquean (ej. tramo fuera del horizonte)
    """
    errores = []
    advertencias = []
    filas_validas = []   # (inicio, fin) de filas completas, para revisar solapes

    for i, fila in tabla.iterrows():
        n = i + 1
        inicio, fin, valor = fila["inicio"], fila["fin"], fila[campo_valor]

        # 1. Campos faltantes
        faltantes = []
        if pd.isna(inicio):
            faltantes.append("inicio")
        if pd.isna(fin):
            faltantes.append("fin")
        if pd.isna(valor):
            faltantes.append(etiqueta_valor)
        if faltantes:
            errores.append(f"Fila {n}: falta completar {', '.join(faltantes)}.")
            continue

        # 2. Fechas invertidas
        if fin <= inicio:
            errores.append(f"Fila {n}: la fecha de fin debe ser posterior a la de inicio.")

        # 3. Valor fuera de rango
        if valor < valor_min or valor > valor_max:
            errores.append(f"Fila {n}: {etiqueta_valor} debe estar entre "
                           f"{valor_min} y {valor_max} (ingresaste {valor:g}).")

        # 4. Tramo fuera del horizonte del pronóstico (advertencia, no error)
        if inicio_horizonte is not None and fin_horizonte is not None:
            def _naive(x):
                t = pd.Timestamp(x)
                return t.tz_localize(None) if t.tzinfo is not None else t
            ini, fn = _naive(inicio), _naive(fin)
            ih, fh = _naive(inicio_horizonte), _naive(fin_horizonte)
            if fn <= ih or ini >= fh:
                advertencias.append(
                    f"Fila {n}: el tramo está fuera del horizonte del pronóstico "
                    f"y no tendrá efecto.")
            elif ini < ih or fn > fh:
                advertencias.append(
                    f"Fila {n}: el tramo se aplicará solo parcialmente "
                    f"(parte queda fuera del horizonte del pronóstico).")

        filas_validas.append((pd.Timestamp(inicio), pd.Timestamp(fin)))

    # 5. Tramos solapados
    for a in range(len(filas_validas)):
        for b in range(a + 1, len(filas_validas)):
            ini_a, fin_a = filas_validas[a]
            ini_b, fin_b = filas_validas[b]
            if ini_a < fin_b and ini_b < fin_a:
                errores.append(f"Las filas {a+1} y {b+1} se solapan en el tiempo.")

    return errores, advertencias


# --- Barra lateral: selección de modo ---
st.sidebar.title("🌬️ WindCast")
st.sidebar.caption("Pronóstico de producción eólica con IA")
modo = st.sidebar.radio("Selecciona un modo:",
                        ["⚙️ Configuración", "📊 Predicción"])

# ============================================
# MODO CONFIGURACIÓN
# ============================================
if modo == "⚙️ Configuración":
    st.title("⚙️ Modo Configuración")
    st.caption("Crea un parque nuevo: define su ubicación y sus grupos de aerogeneradores.")

    # Estado de sesión: lista de grupos que se van añadiendo antes de guardar
    if "grupos_nuevos" not in st.session_state:
        st.session_state.grupos_nuevos = []

    # --- Datos generales del parque ---
    st.subheader("1. Datos del parque")
    g1, g2 = st.columns(2)
    nombre_parque = g1.text_input("Nombre del parque", placeholder="Ej. Mi Parque Eólico")
    pais_parque = g2.text_input("País", placeholder="Ej. México")
    g3, g4 = st.columns(2)
    lat_parque = g3.number_input("Latitud", value=0.0, format="%.5f",
                                 min_value=-90.0, max_value=90.0)
    lon_parque = g4.number_input("Longitud", value=0.0, format="%.5f",
                                 min_value=-180.0, max_value=180.0)

    # --- Añadir un grupo de aerogeneradores ---
    st.subheader("2. Agregar grupo de aerogeneradores")
    with st.form("form_grupo", clear_on_submit=True):
        gc1, gc2, gc3 = st.columns(3)
        g_fabricante = gc1.text_input("Fabricante", placeholder="Ej. Siemens")
        g_modelo = gc2.text_input("Modelo", placeholder="Ej. SWT-2.3-82")
        g_num = gc3.number_input("Nº de turbinas", min_value=1, value=1, step=1)
        gc4, gc5, gc6 = st.columns(3)
        g_buje = gc4.number_input("Altura de buje (m)", min_value=1.0, value=80.0)
        g_pot = gc5.number_input("Potencia nominal (kW)", min_value=1.0, value=2300.0)
        g_diam = gc6.number_input("Diámetro de rotor (m)", min_value=1.0, value=82.0)

        archivo_curva = st.file_uploader(
            "Curva de potencia (CSV o JSON)", type=["csv", "json"])

        agregar = st.form_submit_button("➕ Agregar grupo")

    if agregar:
        # Validar campos del grupo
        if not g_fabricante or not g_modelo:
            st.error("Completa fabricante y modelo del grupo.")
        elif archivo_curva is None:
            st.error("Sube el archivo de la curva de potencia del grupo.")
        else:
            try:
                curva = motor.cargar_curva_desde_archivo(archivo_curva, archivo_curva.name)
                st.session_state.grupos_nuevos.append({
                    "fabricante": g_fabricante,
                    "modelo": g_modelo,
                    "num_turbinas": int(g_num),
                    "altura_buje_m": int(g_buje),
                    "potencia_nominal_kw": int(g_pot),
                    "diametro_rotor_m": int(g_diam),
                    "tipo_curva": "archivo",
                    "curva": curva,
                })
                st.success(f"Grupo agregado: {int(g_num)}× {g_fabricante} {g_modelo} "
                           f"(curva con {len(curva)} puntos).")
            except Exception as e:
                st.error(f"No se pudo leer la curva: {e}")

    # --- Mostrar los grupos añadidos ---
    if st.session_state.grupos_nuevos:
        st.subheader("3. Grupos añadidos")
        for i, gr in enumerate(st.session_state.grupos_nuevos):
            cols = st.columns([5, 1])
            cols[0].write(
                f"**Grupo {i+1}:** {gr['num_turbinas']}× {gr['fabricante']} "
                f"{gr['modelo']} · buje {gr['altura_buje_m']} m · "
                f"{gr['potencia_nominal_kw']} kW · curva {len(gr['curva'])} pts")
            if cols[1].button("🗑️ Quitar", key=f"quitar_{i}"):
                st.session_state.grupos_nuevos.pop(i)
                st.rerun()

        # Potencia instalada total del parque en construcción
        pot_total = sum(g["num_turbinas"] * g["potencia_nominal_kw"]
                        for g in st.session_state.grupos_nuevos) / 1000
        st.info(f"Potencia instalada total: {pot_total:.1f} MW  ·  "
                f"{sum(g['num_turbinas'] for g in st.session_state.grupos_nuevos)} turbinas")

    # --- Guardar el parque ---
    st.subheader("4. Guardar parque")
    puede_guardar = bool(nombre_parque and pais_parque
                         and st.session_state.grupos_nuevos)
    if not puede_guardar:
        st.caption("Completa el nombre, el país y agrega al menos un grupo para guardar.")

    if st.button("💾 Guardar parque", type="primary", disabled=not puede_guardar):
        try:
            # Calcular zona horaria automáticamente desde las coordenadas
            tz = motor.obtener_zona_horaria(lat_parque, lon_parque)

            parque_nuevo = {
                "nombre": nombre_parque,
                "pais": pais_parque,
                "latitud": round(float(lat_parque), 5),
                "longitud": round(float(lon_parque), 5),
                "timezone": tz,
                "grupos": st.session_state.grupos_nuevos,
            }

            # Nombre de archivo seguro a partir del nombre del parque
            base = "".join(c if c.isalnum() else "_" for c in nombre_parque.lower())
            os.makedirs("parques", exist_ok=True)
            ruta = f"parques/parque_{base}.json"
            with open(ruta, "w", encoding="utf-8") as f:
                json.dump(parque_nuevo, f, indent=2, ensure_ascii=False)

            st.success(f"✅ Parque '{nombre_parque}' guardado en {ruta}  "
                       f"(zona horaria detectada: {tz}). "
                       f"Ya está disponible en el Modo Predicción.")
            # Limpiar el estado para empezar otro parque
            st.session_state.grupos_nuevos = []
        except Exception as e:
            st.error(f"No se pudo guardar el parque: {e}")

# ============================================
# MODO PREDICCIÓN
# ============================================
elif modo == "📊 Predicción":
    st.title("📊 Modo Predicción")

    # 1. Seleccionar un parque de los guardados
    parques = listar_parques()
    if not parques:
        st.warning("No hay parques configurados. Ve al Modo Configuración para crear uno.")
        st.stop()

    nombres = {os.path.basename(p): p for p in parques}
    seleccion = st.selectbox("Selecciona un parque:", list(nombres.keys()))
    parque = cargar_parque(nombres[seleccion])

    # 2. Ficha del parque
    pot_instalada_mw = sum(g["num_turbinas"] * g["potencia_nominal_kw"]
                           for g in parque["grupos"]) / 1000
    total_turbinas = sum(g["num_turbinas"] for g in parque["grupos"])

    # Hora actual en la zona horaria del parque (snapshot al cargar/interactuar)
    tz_parque_ficha = parque.get("timezone", "UTC")
    hora_local_parque = pd.Timestamp.now(tz=tz_parque_ficha)

    col1, col2, col3 = st.columns(3)
    col1.metric("Parque", parque["nombre"])
    col2.metric("Turbinas", total_turbinas)
    col3.metric("Potencia instalada", f"{pot_instalada_mw:.1f} MW")
    st.caption(f"📍 Ubicación: {parque['latitud']}, {parque['longitud']}  ·  "
               f"{parque['pais']}  ·  {len(parque['grupos'])} grupo(s)  ·  "
               f"🕐 Hora local del parque: {hora_local_parque.strftime('%d/%m %H:%M')} "
               f"({tz_parque_ficha})")

    # Mapa con la ubicación del parque (marcador tipo pin, desplegado por defecto)
    with st.expander("🗺️ Ver ubicación en el mapa", expanded=True):
        mapa = folium.Map(
            location=[parque["latitud"], parque["longitud"]],
            zoom_start=10, tiles="CartoDB positron")
        folium.Marker(
            location=[parque["latitud"], parque["longitud"]],
            popup=parque["nombre"],
            tooltip=parque["nombre"],
            icon=folium.Icon(color="green", icon="bolt", prefix="fa")
        ).add_to(mapa)
        st_folium(mapa, use_container_width=True, height=400, returned_objects=[])

    # 3. Controles del pronóstico
    st.subheader("Parámetros del pronóstico")

    grupo = parque["grupos"][0]
    num_turbinas = grupo["num_turbinas"]

    # Zona horaria del parque (IANA); maneja el cambio de horario automáticamente
    tz_parque = parque.get("timezone", "UTC")
    ahora_parque = pd.Timestamp.now(tz=tz_parque).tz_localize(None)
    ahora_redondeado = ahora_parque.floor("15min").to_pydatetime()
    # Open-Meteo permite hasta 16 días de pronóstico desde hoy
    max_fin = (ahora_parque.normalize() + pd.Timedelta(days=16)).to_pydatetime()

    st.markdown("**Periodo a pronosticar** (hora local del parque)")
    pc1, pc2, pc3 = st.columns(3)

    # Fecha/hora de inicio: por defecto ahora, no antes de ahora
    f_ini = pc1.date_input("Fecha de inicio", value=ahora_redondeado.date(),
                           min_value=ahora_redondeado.date(),
                           max_value=max_fin.date(), key="f_ini")
    h_ini = pc1.time_input("Hora de inicio", value=ahora_redondeado.time(),
                           step=900, key="h_ini")

    # Fecha/hora de fin: por defecto 7 días después
    fin_defecto = (pd.Timestamp(ahora_redondeado) + pd.Timedelta(days=7)).to_pydatetime()
    f_fin = pc2.date_input("Fecha de fin", value=fin_defecto.date(),
                           min_value=ahora_redondeado.date(),
                           max_value=max_fin.date(), key="f_fin")
    h_fin = pc2.time_input("Hora de fin", value=fin_defecto.time(),
                           step=900, key="h_fin")

    resolucion = pc3.selectbox("Resolución", ["Horaria", "Diaria", "15 min"])

    # Combinar fecha + hora en timestamps
    inicio_periodo = pd.Timestamp.combine(f_ini, h_ini)
    fin_periodo = pd.Timestamp.combine(f_fin, h_fin)

    # Validaciones del periodo
    periodo_valido = True
    if fin_periodo <= inicio_periodo:
        st.error("⚠️ La fecha/hora de fin debe ser posterior a la de inicio.")
        periodo_valido = False
    if inicio_periodo < ahora_parque.floor("15min"):
        st.error("⚠️ La fecha/hora de inicio no puede ser anterior al momento actual.")
        periodo_valido = False

    # Advertencia de calidad: se dispara solo si el fin supera 7 días completos desde hoy.
    # Se usa el final del día 7 (medianoche del día 8) como umbral, para que el
    # periodo por defecto (7 días desde ahora) no la dispare innecesariamente.
    limite_calidad = (ahora_parque.normalize() + pd.Timedelta(days=8))
    if periodo_valido and fin_periodo > limite_calidad:
        dia7 = (ahora_parque.normalize() + pd.Timedelta(days=7)).strftime('%d/%m/%Y')
        st.warning(f"ℹ️ El pronóstico cubre más de 7 días (después del {dia7}). "
                   f"A partir de entonces la confiabilidad del pronóstico "
                   f"meteorológico disminuye.")

    # Días a solicitar a Open-Meteo: desde hoy 00:00 hasta el fin del periodo
    dias_api = int(np.ceil(
        (fin_periodo - ahora_parque.normalize()) / pd.Timedelta(days=1)))
    dias_api = max(1, min(16, dias_api))

    # El horizonte para validar tramos es el periodo elegido por el usuario
    inicio_horizonte = inicio_periodo
    fin_horizonte = fin_periodo
    # Valor por defecto para los controles de tabla
    hoy = inicio_periodo.to_pydatetime()

    # --- Disponibilidad de turbinas ---
    st.markdown("**Disponibilidad de turbinas**")
    modo_disp = st.radio("Modo", ["Fija", "Por tramos"], horizontal=True, key="modo_disp")
    tramos_disp = []
    errores_disp = []
    disponibles = num_turbinas

    if modo_disp == "Fija":
        d1, d2 = st.columns(2)
        tipo_disp = d1.radio("Expresar como", ["Nº de turbinas", "Porcentaje"],
                             horizontal=True, key="tipo_disp")
        if tipo_disp == "Nº de turbinas":
            disponibles = d2.number_input(
                f"Turbinas disponibles (de {num_turbinas})",
                min_value=0, max_value=num_turbinas, value=num_turbinas)
            d2.caption(f"≈ {100 * disponibles / num_turbinas:.0f}%")
        else:
            pct_disp = d2.slider("Turbinas disponibles (%)", 0, 100, 100)
            disponibles = round(num_turbinas * pct_disp / 100)
            d2.caption(f"≈ {disponibles} de {num_turbinas} turbinas")
    else:
        st.caption("Define tramos con distinta disponibilidad. Fuera de los tramos "
                   f"se asume el total ({num_turbinas} turbinas).")
        tabla_disp = st.data_editor(
            pd.DataFrame({"inicio": [hoy], "fin": [hoy + dt.timedelta(hours=4)],
                          "turbinas_disponibles": [num_turbinas]}),
            column_config={
                "inicio": st.column_config.DatetimeColumn("Inicio", step=900),
                "fin": st.column_config.DatetimeColumn("Fin", step=900),
                "turbinas_disponibles": st.column_config.NumberColumn(
                    "Turbinas disponibles", min_value=0, max_value=num_turbinas),
            },
            num_rows="dynamic", key="tabla_disp", use_container_width=True)
        for _, fila in tabla_disp.iterrows():
            if (pd.notna(fila["inicio"]) and pd.notna(fila["fin"])
                    and pd.notna(fila["turbinas_disponibles"])):
                tramos_disp.append({
                    "inicio": pd.Timestamp(fila["inicio"]),
                    "fin": pd.Timestamp(fila["fin"]),
                    "valor": int(fila["turbinas_disponibles"])})
        errores_disp, avisos_disp = validar_tramos(
            tabla_disp, "turbinas_disponibles", 0, num_turbinas, "turbinas",
            inicio_horizonte, fin_horizonte)
        for e in errores_disp:
            st.error("⚠️ Disponibilidad — " + e)
        for a in avisos_disp:
            st.warning("ℹ️ Disponibilidad — " + a)

    # --- Curtailment / restricción de red ---
    st.markdown("**Curtailment (restricción de red)**")
    modo_curt = st.radio("Modo", ["Sin curtailment", "Fijo", "Por tramos"],
                         horizontal=True, key="modo_curt")
    tramos_curt = []
    errores_curt = []
    limite_mw = None

    if modo_curt == "Fijo":
        cc1, cc2 = st.columns(2)
        tipo_limite = cc1.radio("Tipo de límite", ["MW absolutos", "Porcentaje"],
                                horizontal=True, key="tipo_curt")
        if tipo_limite == "MW absolutos":
            limite_mw = cc2.number_input(
                "Límite de potencia (MW)", min_value=0.0,
                max_value=float(pot_instalada_mw), value=float(pot_instalada_mw), step=1.0)
            cc2.caption(f"≈ {100 * limite_mw / pot_instalada_mw:.0f}% de la capacidad")
        else:
            pct = cc2.slider("Límite (% de la capacidad)", 0, 100, 100)
            limite_mw = pot_instalada_mw * pct / 100
            cc2.caption(f"Equivale a {limite_mw:.1f} MW")
    elif modo_curt == "Por tramos":
        st.caption("Define tramos con distinto límite de potencia (MW). Fuera de los "
                   "tramos no hay restricción.")
        tabla_curt = st.data_editor(
            pd.DataFrame({"inicio": [hoy], "fin": [hoy + dt.timedelta(hours=4)],
                          "limite_mw": [pot_instalada_mw / 2]}),
            column_config={
                "inicio": st.column_config.DatetimeColumn("Inicio", step=900),
                "fin": st.column_config.DatetimeColumn("Fin", step=900),
                "limite_mw": st.column_config.NumberColumn(
                    "Límite (MW)", min_value=0.0, max_value=float(pot_instalada_mw)),
            },
            num_rows="dynamic", key="tabla_curt", use_container_width=True)
        for _, fila in tabla_curt.iterrows():
            if (pd.notna(fila["inicio"]) and pd.notna(fila["fin"])
                    and pd.notna(fila["limite_mw"])):
                tramos_curt.append({
                    "inicio": pd.Timestamp(fila["inicio"]),
                    "fin": pd.Timestamp(fila["fin"]),
                    "valor": float(fila["limite_mw"])})
        errores_curt, avisos_curt = validar_tramos(
            tabla_curt, "limite_mw", 0, pot_instalada_mw, "límite (MW)",
            inicio_horizonte, fin_horizonte)
        for e in errores_curt:
            st.error("⚠️ Curtailment — " + e)
        for a in avisos_curt:
            st.warning("ℹ️ Curtailment — " + a)

    # 4. Botón para generar el pronóstico (bloqueado si hay errores o periodo inválido)
    hay_errores = bool(errores_disp or errores_curt) or not periodo_valido
    if errores_disp or errores_curt:
        st.warning("Corrige los errores marcados antes de generar el pronóstico.")

    if st.button("🔮 Generar pronóstico", type="primary", disabled=hay_errores):
        with st.spinner("Consultando Open-Meteo y calculando producción..."):
            # Obtener viento a máxima resolución, interpolado a altura de buje
            df_viento = motor.obtener_pronostico_viento(
                lat=parque["latitud"], lon=parque["longitud"],
                altura_buje=grupo["altura_buje_m"], dias=dias_api)
            # Recortar al periodo elegido por el usuario
            df_viento = motor.recortar_rango(df_viento, inicio_periodo, fin_periodo)

            # Producción base por turbina (la disponibilidad se aplica después)
            curva = np.array(grupo["curva"])
            df_prod = motor.calcular_produccion(
                df_viento, curva[:, 0], curva[:, 1], turbinas_disponibles=1)

            # Aplicar disponibilidad (fija o por tramos)
            if modo_disp == "Por tramos" and tramos_disp:
                df_prod = motor.aplicar_disponibilidad_tramos(
                    df_prod, num_turbinas, None,
                    disponible_defecto=num_turbinas, tramos=tramos_disp)
            else:
                df_prod["produccion_parque_kw"] = df_prod["potencia_turbina_kw"] * disponibles
                df_prod["produccion_parque_mw"] = df_prod["produccion_parque_kw"] / 1000

            # Aplicar curtailment (fijo o por tramos)
            if modo_curt == "Por tramos" and tramos_curt:
                df_prod = motor.aplicar_curtailment_tramos(df_prod, None, tramos_curt)
            elif modo_curt == "Fijo":
                df_prod = motor.aplicar_curtailment(df_prod, limite_mw)

            # Agregar a la resolución pedida
            freq = {"Horaria": "h", "Diaria": "D", "15 min": "15min"}[resolucion]
            df_final = motor.agregar_resolucion(df_prod, freq)

        # 5. Mostrar resultados
        st.success(f"Pronóstico generado: {len(df_final)} registros")

        paso_horas = {"h": 1.0, "D": 24.0, "15min": 0.25}[freq]
        energia = df_final["produccion_parque_mw"].sum() * paso_horas
        m1, m2, m3 = st.columns(3)
        m1.metric("Producción media", f"{df_final['produccion_parque_mw'].mean():.2f} MW")
        m2.metric("Producción máxima", f"{df_final['produccion_parque_mw'].max():.2f} MW")
        m3.metric("Energía total", f"{energia:.0f} MWh")

        # 5b. Gráfica de producción con marcas de afectación (Plotly)
        st.subheader("Producción pronosticada")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_final["tiempo"], y=df_final["produccion_parque_mw"],
            mode="lines", name="Producción (MW)",
            line=dict(color="#21295C", width=2),
            fill="tozeroy", fillcolor="rgba(2,195,154,0.15)"))
        fig.add_hline(y=pot_instalada_mw, line_dash="dot", line_color="#065A82",
                      annotation_text=f"Potencia instalada ({pot_instalada_mw:.0f} MW)",
                      annotation_position="top left")

        # Marcas de disponibilidad reducida (naranja)
        pot_nominal_turbina_mw = grupo["potencia_nominal_kw"] / 1000
        if modo_disp == "Fija" and disponibles < num_turbinas:
            techo_mw = disponibles * pot_nominal_turbina_mw
            fig.add_hline(
                y=techo_mw, line_dash="dash", line_color="orange",
                annotation_text=f"Máx. con {disponibles}/{num_turbinas} turbinas "
                                f"({techo_mw:.0f} MW)",
                annotation_position="top right")
        else:
            for t in tramos_disp:
                if t["valor"] < num_turbinas:
                    fig.add_vrect(
                        x0=t["inicio"], x1=t["fin"],
                        fillcolor="orange", opacity=0.15, line_width=0,
                        annotation_text=f"Disp. {t['valor']}/{num_turbinas}",
                        annotation_position="top left", annotation_font_size=10)
                    techo_mw = t["valor"] * pot_nominal_turbina_mw
                    fig.add_shape(type="line", x0=t["inicio"], x1=t["fin"],
                                  y0=techo_mw, y1=techo_mw,
                                  line=dict(color="orange", width=1.5, dash="dash"))

        # Bandas de curtailment (rojo) + línea del límite por tramo
        for t in tramos_curt:
            fig.add_vrect(
                x0=t["inicio"], x1=t["fin"],
                fillcolor="red", opacity=0.10, line_width=0,
                annotation_text=f"Curt. {t['valor']:.0f} MW",
                annotation_position="bottom left", annotation_font_size=10)
            fig.add_shape(type="line", x0=t["inicio"], x1=t["fin"],
                          y0=t["valor"], y1=t["valor"],
                          line=dict(color="red", width=1.5, dash="dash"))

        # Si el curtailment es fijo y restrictivo, marcar el límite global
        if modo_curt == "Fijo" and limite_mw is not None and limite_mw < pot_instalada_mw:
            fig.add_hline(y=limite_mw, line_dash="dash", line_color="red",
                          annotation_text=f"Límite curtailment ({limite_mw:.0f} MW)",
                          annotation_position="bottom right")

        fig.update_layout(
            xaxis_title="Fecha y hora", yaxis_title="Producción del parque (MW)",
            height=420, margin=dict(t=30, b=40), hovermode="x unified",
            showlegend=False,
            xaxis_range=[df_final["tiempo"].min(), df_final["tiempo"].max()])
        st.plotly_chart(fig, use_container_width=True)

        # 5c. Gráfica de viento (Plotly)
        st.subheader("Viento a altura de buje")
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(
            x=df_final["tiempo"], y=df_final["viento_buje"],
            mode="lines", name="Viento (m/s)",
            line=dict(color="#1C7293", width=2),
            fill="tozeroy", fillcolor="rgba(28,114,147,0.12)"))
        fig_v.add_hline(y=3, line_dash="dot", line_color="gray",
                        annotation_text="Arranque (~3 m/s)",
                        annotation_position="bottom left")
        fig_v.update_layout(
            xaxis_title="Fecha y hora", yaxis_title="Viento a altura de buje (m/s)",
            height=320, margin=dict(t=20, b=40), hovermode="x unified",
            showlegend=False,
            xaxis_range=[df_final["tiempo"].min(), df_final["tiempo"].max()])
        st.plotly_chart(fig_v, use_container_width=True)

        # 5d. Tabla de datos + descarga CSV
        with st.expander("Ver datos en tabla"):
            st.dataframe(df_final)

        # Preparar el CSV para descarga
        df_export = df_final.copy()
        # Redondear para un archivo más limpio
        for col in ["viento_buje", "potencia_turbina_kw", "produccion_parque_mw"]:
            if col in df_export.columns:
                df_export[col] = df_export[col].round(3)
        csv = df_export.to_csv(index=False).encode("utf-8")

        # Nombre de archivo con parque y fecha
        nombre_archivo = (f"windcast_{parque['nombre'].replace(' ', '_')}_"
                          f"{inicio_periodo.strftime('%Y%m%d')}.csv")

        st.download_button(
            label="📥 Descargar pronóstico (CSV)",
            data=csv,
            file_name=nombre_archivo,
            mime="text/csv")