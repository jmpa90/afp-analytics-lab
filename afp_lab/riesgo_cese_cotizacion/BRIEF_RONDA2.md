# Brief 04 · Cese — ronda 2 (addendum al BRIEF.md)

**Responsable**: DS-3, después del módulo de planilla regulatoria y antes del proyecto 06.

1. `run.py --perfil {sistema,cuprum}`; correr ambos y comparar (tasas de cese y reingreso, AUC, Brier,
   error del agregado). Cuprum: contratos más estables, más rotación ejecutiva y jubilación, densidad 67 %.
2. Si `afp_lab/common/planilla.py` ya existe, usar los códigos de movimiento de personal de la planilla
   (2 retiro, 3 subsidio, 4 permiso sin goce, 7/8 contrato) como features con el mismo desfase que la
   RIM, y medir cuánto aportan frente a los eventos AFC.
3. README con sección "Ronda 2". Sin tocar nada fuera de la carpeta.
