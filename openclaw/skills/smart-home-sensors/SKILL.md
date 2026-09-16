---
name: smart-home-sensors
description: Consulta datos simulados del reloj para responder preguntas sobre sueño, pasos y distancia.
---

Para preguntas sobre actividad, sueño, pasos o distancia, usa la herramienta
`query_aggregation` antes de responder. No inventes resultados.

Tipos disponibles:

- `watch.sleep`: `total` devuelve minutos dormidos; `mean` devuelve la fracción de sueño del intervalo.
- `watch.steps`: pasos por minuto; usa `total` para el total de pasos.
- `watch.distance_m`: distancia en metros por minuto; usa `total` para la distancia total.

El dato tiene resolución de un minuto y los intervalos son `[time_init, time_end)`.
Para «¿he dormido bien?» en este único día simulado, consulta `watch.sleep`
con agregación `total` entre `00:00` y `24:00`. Explica que la simulación solo
estima duración, no calidad clínica del sueño.
