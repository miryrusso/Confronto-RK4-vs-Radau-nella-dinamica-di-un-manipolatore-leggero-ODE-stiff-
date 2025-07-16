
# REPORT PROGETTO: Stabilità RK4 vs Radau

## Obiettivi del Progetto
- Confrontare la stabilità di metodi numerici espliciti (RK4) vs impliciti (Radau)
- Analizzare l'effetto della rigidezza (stiffness) su sistemi ODE
- Valutare accuratezza, efficienza e conservazione dell'energia

## Metodologia
1. Modello matematico: manipolatore planare 2-R con molle torsionali
2. Tre livelli di rigidezza: k = 10, 100, 500 N⋅m/rad
3. Cinque metodi numerici testati:
   - Euler esplicito/implicito
   - Metodo del trapezio
   - Runge-Kutta 4 (RK4)
   - Radau IIA

## Metriche di Valutazione
- Errore RMS rispetto a soluzione di riferimento
- Tempo di calcolo CPU
- Numero di valutazioni della funzione ODE
- Drift energetico (conservazione dell'energia)
- Regioni di stabilità

## File Generati
- results_manipulator.csv: dati numerici completi
- error_vs_step.png: accuratezza vs passo
- efficiency_analysis.png: errore vs tempo CPU
- energy_conservation.png: drift energetico
- stability_regions.png: mappa di stabilità
- joint_trajectories.png: traiettorie angoli
- phase_portraits.png: ritratti di fase
- end_effector_trajectory.png: traiettorie cartesiane
- energy_evolution.png: evoluzione energia
- stability_comparison.png: confronto stabilità
- manipulator_animation.gif: animazione (se creata)

## Conclusioni Attese
- RK4: alta accuratezza per problemi non-stiff, instabile con passi grandi per k elevati
- Radau: stabile per tutti i valori di k, gestisce la stiffness efficacemente
- Trade-off accuratezza/costo computazionale dipendente dalla rigidezza del sistema
