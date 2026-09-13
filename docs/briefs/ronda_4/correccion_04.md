# Brief — corrección de `afp_lab/riesgo_cese_cotizacion` tras la calibración de densidad (ronda 3)

## Problema
Al calibrar la densidad del perfil `sistema` a la Ficha 165 (`p_activo_inicial` 0,78→0,45,
`factor_contratacion` 1,0→0,20 en `afp_lab/common/synth_base.py`), el AUC de **reingreso** del
proyecto 04 cayó de 0,720 a 0,605 (semilla 42, 20 000 afiliados), bajo su umbral de 0,70. El
AUC de cese se mantuvo. Ver `docs/revisiones/ronda_3/empleador.md` y `afp_lab/common/README.md`.

## Pedido
1. Diagnosticar la causa con evidencia: ¿el target de reingreso se volvió más raro o más
   aleatorio (menos contrataciones con `factor_contratacion` 0,20)? ¿Las features de historial
   laboral perdieron poder? ¿El horizonte de reingreso quedó corto para la nueva dinámica?
   Medir prevalencia del target, importancia de features y AUC por tramo de duración del cese,
   antes y después (el "antes" se reproduce con monkeypatch de `PERFILES`, sin tocar common).
2. Corregir en 04 sin bajar el umbral y sin tocar common: candidatos — features de historial
   (número de episodios de cese, duración media, meses desde el último reingreso, rubro y
   tamaño del último empleador, estacionalidad de contratación del rubro), horizonte de
   reingreso, o el target (definir reingreso con al menos un mes de cotización efectiva).
   Si el problema es que la dinámica del generador dejó el reingreso casi impredecible, decirlo
   con evidencia y proponer el ajuste mínimo al generador para la próxima ronda, sin aplicarlo.
3. Comparar sistema y cuprum, tests, README de 04 (sección "Ronda 3 — corrección").

## Restricciones
Escala 20 000 afiliados con `n_jobs=2`, corridas < 3 min, tests de 04 < 60 s, sin suite completa,
sin git. Presupuesto: ~150 llamadas a herramientas y ~60 minutos.
