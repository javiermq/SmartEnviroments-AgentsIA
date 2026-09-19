# Datos de contexto

- `user_context.json`: perfil editable de Mariola; se vuelve a leer en cada turno.
- `agent_style.txt`: instrucciones editables de personalidad, humor y acompañamiento;
  también se recargan en cada turno. Se mantienen separadas de los datos del usuario.
- `simulated_sensor_data_YYYY-MM-DD.tsv`: una muestra por minuto.
- `simulated_activities_YYYY-MM-DD.tsv`: actividades con intervalos [inicio, fin).

El 15 de septiembre de 2026 es una copia sintética del día 16 desplazada un día
hacia atrás, incluidos los finales de actividad que cruzan medianoche. Permite
probar ventanas nocturnas; no representa observaciones reales adicionales.

Las antiguas columnas `uwb_distance_X_m` son ahora `cercania_distance_X`:
`1 - min(10, max(0, distancia_en_metros)) / 10`. Son adimensionales y están
redondeadas a tres decimales: 1 significa distancia cero y 0 significa 10 m o más.
Las celdas vacías se conservan como desconocidas y llegan al prompt como `null`.

El prompt usa la muestra del minuto de t0 (nunca una posterior), las actividades
en curso y el historial que interseca [t0 - 12 horas, t0). Los intervalos históricos
se recortan a esa ventana; no se envía el final futuro de la actividad actual.
Si no hay datos para ese instante, se indica su ausencia.

Se cargan todos los TSV que coincidan con esos patrones. Las rutas se resuelven
desde el código, independientemente del directorio desde el que se ejecute Python.
En las consultas del agente, HH:MM se refiere al día de t0. En la CLI de agregación,
sin t0, se refiere al último día de datos. Para varios días, usar ISO-8601 con zona.
El t0 de una sesión es fijo; iniciar otra sesión para consultar otro instante.
