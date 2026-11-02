# Agente histórico CAREBOT con Qwen

El agente usa cuatro herramientas de métricas objetivas y una herramienta de
reglas HAR explícitas. Las observaciones, los resultados de reglas y la explicación
del LLM permanecen separados. No se modifican el backend anterior, HERA ni los TSV.

## Ejecución

Requiere Python 3.11+, la zona Europe/Madrid disponible y el Ollama existente.
No se han instalado dependencias ni modelos. Si otro entorno Windows carece de
zonas horarias, instalar `tzdata` en ese entorno. Se comprobó el modelo local
`qwen3:4b-instruct-2507-q4_K_M`; se reutiliza `Settings.from_env` y su API de chat.

Desde la raíz del proyecto:

```powershell
python -m smart_home_agent.historical_agent --summary --t0 2026-10-31T08:30:00+01:00
python -m smart_home_agent.historical_agent --activities all --t0 2026-11-02T12:00:00+01:00
python -m smart_home_agent.historical_agent --har-rules docs/har_rules.example.json "¿Cuándo parece que usé la cocina hoy?"
python -m smart_home_agent.historical_agent "¿Cuántos pasos hay observados en las últimas 24 horas?"
python -m unittest tests.test_historical_har tests.test_conversation_backend -q
python -m scripts.evaluate_historical_agent
```

Opciones: `--data DIRECTORIO`, `--model MODELO_INSTALADO`, `--max-calls 6`,
`--audit temp_data/har_audit.jsonl`. Solo el operador elige directorios y reglas;
Qwen no puede seleccionar rutas, modificar umbrales o ejecutar código arbitrario.
El registro JSONL guarda pregunta, referencia temporal, modelo, hashes de fuentes,
llamadas, motivos, parámetros, resultados y respuesta final o error. Contiene
información personal y se guarda por defecto en `temp_data`, excluido de Git.

Entrada local comprobada: tres TSV en `dataset_home_real/mariola`, del 31/10 al
02/11/2026, con 4.041 filas y 25 columnas. No contienen anotaciones HAR. No se
presume que sean datos simulados: la documentación de HERA describe el export
experimental. Los datos locales ya estaban sin seguimiento de Git; no se añaden
al repositorio mediante este trabajo.

## Herramientas

```python
from smart_home_agent.historical_har import History
from smart_home_agent.har_rules import HarRules

h = History("dataset_home_real/mariola", har_rules=HarRules(merge_gap=3))
summary = h.resumir_intervalo(window="ultimas_24h")
records = h.consultar_minutos(["habitacion", "entrada_binario7"], limit=30)
segments = h.buscar_episodios(
    [{"column": "habitacion", "op": "eq", "value": "Cocina"}], merge_gap=3)
hypotheses = h.inferir_actividades(activity="cocinar")
trend = h.analizar_tendencia("watch_01.hr_bpm", window_minutes=10, min_coverage=.8)
```

- `resumir_intervalo`: suma observada de pasos; media, mínimo y máximo de HR,
  lux, humedad y potencia; tiempos observados por habitación y transiciones entre
  minutos consecutivos conocidos; minutos binarios con valor 1 y recuentos de red.
  Siempre ofrece cobertura por señal. W no se suman como energía. HR representa
  la última lectura de cada minuto, no todas las mediciones originales.
- `consultar_minutos`: valores originales, vacíos como `null`, unidades,
  procedencia, número total de registros y paginación de hasta 120 filas.
- `buscar_episodios`: tramos por condiciones objetivas `eq/ne/gt/ge/lt/le`, con
  `all` o `any`. Habitación solo acepta `eq/ne` con texto exacto. Los valores de
  condiciones numéricas deben ser números JSON sin comillas. Por defecto no une
  huecos. Devuelve `matching_minutes`, `unknown_minutes`, `nonmatching_minutes`
  y duración global. Las condiciones emplean lógica con desconocidos: un falso
  conocido basta para descartar `all`; un verdadero conocido basta para `any`.
- `inferir_actividades`: capa separada de reglas configurables. Devuelve etiquetas
  de hipótesis, regla aplicada, parámetros, cobertura de regla, muestra de evidencia
  con procedencia y advertencias. La muestra está identificada como parcial cuando
  el episodio tiene más de cuatro minutos de evidencia; el resto se puede consultar
  con las herramientas objetivas. Los umbrales no están calibrados individualmente.
- `analizar_tendencia`: pendiente lineal sobre timestamps observados, cambio entre
  primera/última lectura y diferencia de medias de dos ventanas consecutivas. La
  ventana es un entero de 2 a 1.440 minutos. No interpola; exige al menos dos
  lecturas y, por defecto, cobertura del 80% en cada ventana para sus tendencias.
  Si una ventana no cumple, su pendiente/cambio son `null`; la comparación de medias
  requiere que ambas cumplan. Las medias observadas siguen disponibles con cobertura.

Todas aceptan `t0`. Las consultas de intervalo aceptan `window` (`hoy`, `ayer`,
`ultimas_24h`) o `start/end` explícitos. La tendencia usa `window_minutes` y termina
en `t0`: ventana actual [t0-W,t0) y previa [t0-2W,t0-W).
Usan minutos completos e intervalos semiabiertos. Sin `t0`, la referencia es el
final del último minuto exportado, nunca el reloj del ordenador. La fila 08:30
representa [08:30,08:31) y no entra en una consulta que termina a las 08:30.
Las fechas deben incluir el desfase correcto de Europe/Madrid, para distinguir
horas repetidas y rechazar horas inexistentes. Los días naturales pueden tener
23 o 25 horas; las últimas 24 horas son siempre 1.440 minutos transcurridos.

## Reglas HAR operativas

Los valores iniciales están en `docs/har_rules.example.json` y `HarRules`.

| Etiqueta | Evidencia requerida | Límite |
|---|---|---|
| cocinar | Habitación Cocina y frigo >=10 lx o vasos/platos=1 en ese minuto | Uso compatible; no prueba preparación de comida |
| bano | Habitación WC/Baño/Bano | Localización estimada, no actividad validada |
| ducha | Baño observado y subida >=8 puntos porcentuales respecto a lectura conocida anterior al tramo de baño, con referencia de hasta 60 min | Admite hasta 10 min de retraso; humedad alta aislada no basta |
| bedtime | Modo del SO=1 | No demuestra sueño ni despertar |
| descansar | Pasos observados=0 y HR observada <=65 bpm durante al menos 10 min consecutivos | Vacíos interrumpen la racha; umbral provisional |
| ejercicio | Pasos observados >=60 por minuto y HR >=100 bpm, ambos durante al menos 5 min consecutivos | Actividad física ambulatoria compatible; no confirma deporte ni detecta necesariamente ejercicio sin pasos |
| salir | Sensor entrada=1 | Evento compatible con salida o entrada; sin dirección ni duración fuera |

La hipótesis de ducha se sitúa en el último minuto de baño observado, conservando
los tiempos de lectura de humedad y referencia por separado. La evidencia posterior
solo se usa si pertenece al intervalo consultado y no supera `t0`; por ello una
consulta posterior puede aportar evidencia que antes faltaba. El minuto de soporte
no es una estimación de la duración real de ducha.

La unión de episodios HAR admite por defecto huecos de hasta 3 min, con un máximo
acumulado de 9 min. Se distingue tiempo desconocido de interrupciones observadas.
En cocina, una visita al WC puede quedar dentro del intervalo global, pero nunca
suma minutos de cocinar. Evidencia contradictoria, como Bedtime=0 entre dos
activaciones o movimiento entre tramos de descanso, impide la unión. Una fracción
de huecos/interrupciones superior al 40% se advierte explícitamente. Los eventos de
entrada solo agrupan minutos contiguos; nunca se prolongan hasta otra apertura.

Ducha y Baño pueden solaparse. Los totales de soporte por etiqueta no son exclusivos:
para sumar actividades se debe acordar prioridad o usar la unión lógica de minutos.
La duración global de un episodio incluye huecos; `evidence_minutes` no los cuenta.
En las herramientas objetivas el campo equivalente es `matching_minutes`.

### Tendencias y ejercicio

La pendiente se calcula por mínimos cuadrados: sum((ti-media(t))*(xi-media(x))) /
sum((ti-media(t))²), usando tiempos transcurridos reales en minutos y solo las
muestras observadas. La diferencia entre primera y última usa sus timestamps;
en una ventana completa de 10 minutos, sus inicios están separados por 9 minutos.
El cambio entre medias de ventanas se devuelve en unidades de la señal, sin
dividirlo por una duración que podría no representar sus muestras cuando hay huecos.

HR tiene pendiente en bpm/min; humedad en puntos porcentuales/min. Los pasos son
incrementos por minuto: su nivel representa cadencia (pasos/min), y su pendiente
el cambio de cadencia (pasos/min/min), no crecimiento de un contador acumulado.
No se calcula derivada de binarios ni códigos de red.

Un incremento puede ayudar a explorar inicio o recuperación de una actividad;
no identifica por sí solo ejercicio. La regla de ejercicio no exige pendiente
positiva: permite niveles altos estables. Los valores 60 pasos/min, 100 bpm y
5 min son parámetros experimentales, configurables con `--har-rules`, sin
calibración individual ni significado clínico. Caminar rápido u otras tareas
pueden cumplirlos; ciclismo o fuerza pueden no generar pasos. Medir si estas
características mejoran HAR requiere anotaciones y comparación con/sin tendencia.

Para delimitar ejercicio, Qwen debe obtener primero los episodios de la regla y
contrastar sus límites con tendencias de pasos y HR en ventanas cercanas al inicio
y final. Una ventana termina en su `t0`: para abarcar una transición puede acabar
después del límite candidato, siempre sin superar el t0 absoluto de la pregunta.
Se conserva separado el intervalo de soporte de regla de una posible fase de
subida o recuperación; la tendencia no prolonga automáticamente el episodio. Si
el episodio llega al final de la consulta, su final está censurado y puede continuar.

Qwen tiene instrucciones de consultar también `analizar_tendencia` para preguntas
de ducha o evolución de humedad, con `column="bano_humedad7_pct"` y una ventana
configurable (por ejemplo, 10 minutos). Si conoce un instante candidato, puede
examinar una ventana pasada con `t0` anterior al límite de la pregunta. La pendiente
en puntos porcentuales/min se contrasta con la referencia previa y la presencia
estimada en baño. No sustituye la regla de ducha. Como el sensor registra cambios,
la cobertura puede ser insuficiente: en ese caso se declara indeterminado, sin
interpolar ni reducir artificialmente los requisitos de cobertura.

## Entrada, procedencia y preprocesamiento

El export de HERA se inspeccionó solo en lectura, incluyendo
`docs/EXPORT-EXPERIMENTO-POR-MINUTO.md`, `fleet/sensors_per_minute.py` y
`fleet/user_per_minute.py`. El primero documenta los máximos ambientales observados
por minuto y la omisión de indicadores de retenidos en el export reducido. La
conversión binaria es on/true=1 y off/false=0. Esto no identifica por sí solo una
apertura física ni su dirección. Red: 1 contacto, 2 reposo, 3 desconexión explícita,
4 Wi-Fi sin verificar. Ninguno de estos códigos se usa para la regla de salir.

El lector valida sin reparar: esquema, reloj `watch_01`, timestamps coherentes,
valores finitos y minutos únicos. Duplicados o revisiones competidoras se rechazan
para resolverlos en preprocesamiento externo. Se preservan store, revisión,
archivo, línea y hashes. No hay interpolación ni arrastre de estados.

Cada pregunta usa un snapshot en memoria. Se comprueban tamaño y fecha de los
archivos antes y después de leer, con hasta tres intentos. No bloquea exportadores.
Es una lectura optimista, no una transacción entre archivos del exportador.
`t0` restringe tiempo de observación; no reconstruye las revisiones que estaban
históricamente disponibles en ese momento.

### Feedback para mejorar la entrada

En el 02/11 entre 00:00 y 12:00 hay 720 filas, pero solo 314 minutos con pasos,
196 con HR y 149 con habitación estimada. Los 149 de habitación son Dormitorio.
Hay observaciones ambientales en cocina, pero sin coincidencia con localización
Cocina: no hay soporte suficiente para la regla de cocinar en esa ventana. Esto
es falta de evidencia multimodal, no prueba de que no se cocinara.

Mejoras propuestas para un preprocesamiento separado, **aún no implementado**:

1. Añadir por sensor conteo de observaciones, retenidos, instante de evento y
   recepción. Permitiría distinguir una activación nueva de un valor retenido.
2. Conservar dos vistas: eventos observados y, solo cuando la semántica lo permita,
   estado conocido con tiempo desde última observación, límite de vigencia y estado
   del registrador. No convertir un vacío en cero ni propagar valores indefinidamente.
3. Exportar para humedad una referencia conocida, cambio en puntos porcentuales y
   tiempo transcurrido, conservando ambos timestamps. Son características objetivas,
   no una etiqueta de ducha ni muestras interpoladas.
4. Mantener la base temporal y proyección de reloj. En 3.635 filas con source_minute,
   este es un minuto mayor que timestamp_unix/60. El exportador distingue minuto
   fuente y minuto central proyectado: no desplazar las señales automáticamente.
   Incorporar clock_basis/central_minute ayudaría a auditar esa correspondencia.
5. Preparar ventanas multimodales con cobertura por señal, conteos de eventos y
   procedencia. Añadir anotaciones humanas y asociaciones verificadas entre persona,
   reloj y UWB antes de medir precisión HAR o personalizar umbrales.

## Pruebas y estado de Qwen

Se ejecutaron 34 pruebas: 31 del módulo nuevo y 3 del backend anterior. Cubren
medianoche, horario de verano/invierno, t0, ceros/vacíos, estadísticas, duplicados,
paginación, condiciones, huecos 2/3/4, límites acumulados, interrupciones por baño,
humedad alta sin subida, ausencia de referencia, subida con retraso, límites de
retraso/referencia, Bedtime, descanso con información insuficiente y entrada sin
red. Hay pruebas de ejecución de herramientas, presupuesto y comparación con
lectura CSV independiente de los TSV privados. Esa integración se omite si los
datos locales no están disponibles.
Las pruebas de tendencia incluyen crecimiento/descenso lineal, señal constante,
huecos, cobertura insuficiente, cambio horario y restricción t0. Las de ejercicio
incluyen niveles altos estables, datos ausentes, umbrales y duración mínima.
Se comprueba también un intervalo de ejercicio con inicio/final observados y
recuperación posterior, conservando la separación entre regla y tendencia.

Las pruebas antiguas `test_context` y `test_verified_metrics` presentan errores de
importación: solicitan `_system_prompt` y `_run_turn` ausentes de conversation_demo.
No se han modificado esos módulos.

Se ejecutaron las seis preguntas de ejemplo con Ollama. El cálculo de pasos
coincidió con una lectura CSV independiente y declaró cobertura parcial. Las
pruebas iniciales sin reglas mostraron errores semánticos sobre Bedtime, humedad y
paginación, incluso tras reforzar instrucciones. Se añadieron después las reglas
explícitas y una nueva ejecución se registra en `temp_data/har_evaluation/`.
Las explicaciones del modelo requieren revisión: el funcionamiento de herramientas
y las reglas deterministas no demuestran fiabilidad completa de las respuestas
ni precisión HAR medida.

En la repetición final de la consulta sobre tiempo fuera, Qwen consultó las reglas,
presentó los eventos de entrada observados y mantuvo el tiempo fuera indeterminado.
Se comprobó también que una petición errónea de un intervalo posterior a t0 se
rechaza; el agente dispone de un recordatorio de corrección antes de cerrar sin
evidencia válida. Persisten posibles imprecisiones en las explicaciones del modelo,
por lo que los resultados numéricos y las reglas auditadas son la referencia.

## Banco de 100 preguntas de este módulo

```powershell
python -m scripts.build_har_qa
```

Genera en `temp_data/har_qa_100/` un documento HTML, texto de lectura, referencias
JSONL y metadatos con hashes/reglas. Hay 33 casos por día y una comparación global,
con 21 horas de referencia distintas en la muestra comprobada. Las preguntas se
limitan a actividades, sus señales de apoyo, duración, cobertura y evolución.
Cada registro incluye pregunta, t0, intervalo, respuesta y hechos esperados.

Las referencias se calculan con las herramientas y reglas, no mediante 100 llamadas
a Qwen ni con anotaciones humanas. Para evaluar el agente, darle solo `question`
y `t0`; conservar respuesta/hechos esperados fuera de su contexto. Comparar
cálculos, evidencia y límites, sin exigir coincidencia literal ni presentar el
resultado como precisión HAR. Los casos de días cerrados con «hoy» se cortan a las
23:59; la comparación global incluye todos los minutos exportados. Se verificaron
100 casos únicos, ausencia de evidencia temporal posterior a cada corte y 12
sumas de pasos mediante lectura CSV independiente.
