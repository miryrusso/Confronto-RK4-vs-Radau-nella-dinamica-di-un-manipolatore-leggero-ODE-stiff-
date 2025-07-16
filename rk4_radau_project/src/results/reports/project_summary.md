# REPORT PROGETTO: Stabilità RK4 vs Radau

## 📊 Risultati Generati

### Grafici (9 file)
- `efficiency_analysis.png`
- `end_effector_trajectory.png`
- `energy_conservation.png`
- `energy_evolution.png`
- `error_vs_step.png`
- `joint_trajectories.png`
- `phase_portraits.png`
- `stability_comparison.png`
- `stability_regions.png`

### Dati (1 file)
- `manipulator_results.csv`

## 🎯 Struttura dei Risultati

```
results/
├── graphs/          # Grafici di analisi e traiettorie
├── data/           # Dati numerici CSV
├── animations/     # Animazioni GIF (opzionali)
└── reports/        # Questo report
```

## 🔬 Metodi Analizzati

1. **Euler Esplicito** (ordine 1, esplicito)
2. **Euler Implicito** (ordine 1, A-stabile, L-stabile)
3. **Metodo del Trapezio** (ordine 2, A-stabile)
4. **Runge-Kutta 4** (ordine 4, esplicito) - PROTAGONISTA
5. **Radau IIA** (ordine 5, A-stabile, L-stabile) - PROTAGONISTA

## 🎓 Conclusioni Attese

- **RK4**: Ottima accuratezza per problemi non-stiff, instabile per k grandi
- **Radau**: Sempre stabile, ideale per problemi stiff
- **Trade-off**: accuratezza vs stabilità vs costo computazionale

## 📚 Files per la Relazione

Tutti i grafici sono pronti per essere inclusi nella relazione finale.
I dati numerici sono disponibili in formato CSV per analisi aggiuntive.

---
Generato automaticamente da save_results.py
