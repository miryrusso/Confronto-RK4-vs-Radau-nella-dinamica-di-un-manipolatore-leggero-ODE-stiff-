"""
main.py - Script principale per l'analisi della stabilità RK4 vs Radau

Esegue l'intera campagna di test per confrontare i metodi numerici
su un manipolatore planare 2-R con diverse rigidezze.
"""

import sys
import os
import time
from pathlib import Path
import argparse

# Import dei moduli del progetto
from dynamics import ManipulatorDynamics
from integrators import NumericalIntegrators
from experiments import ExperimentManager
from trajectory_visualization import TrajectoryVisualizer

def print_header():
    """Stampa l'intestazione del progetto."""
    print("=" * 80)
    print("STABILITÀ RK4 vs RADAU NELLA DINAMICA DI UN MANIPOLATORE LEGGERO")
    print("Progetto Finale - Analisi Numerica")
    print("Università degli Studi di Catania")
    print("=" * 80)
    print()

def test_installation():
    """Verifica che tutti i moduli e dipendenze siano installati correttamente."""
    print(" Verifica installazione...")
    
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd
        import seaborn as sns
        from scipy.integrate import solve_ivp
        print("  ✓ Tutti i pacchetti richiesti sono installati")
        
        # Test delle classi principali
        robot = ManipulatorDynamics()
        integrator = NumericalIntegrators(robot.manipulator_ode)
        print("  ✓ Classi principali inizializzate correttamente")
        
        # Test rapido del sistema
        y0 = np.array([0.1, 0.1, 0.0, 0.0])  # Assicurati che sia numpy array
        k = 100.0
        ode_result = robot.manipulator_ode(0.0, y0, k)
        print(f"  ✓ Test ODE: f(0, y0) = {ode_result[:2]}")
        
        return True
        
    except ImportError as e:
        print(f"  ✗ Errore di importazione: {e}")
        print("\nInstalla i pacchetti mancanti con:")
        print("pip install numpy matplotlib pandas seaborn scipy")
        return False
    
    except Exception as e:
        print(f"  ✗ Errore durante il test: {e}")
        return False

def run_quick_demo():
    """Esegue una demo rapida per verificare il funzionamento."""
    print(" Demo rapida del sistema...")
    
    # Test con un caso semplice
    robot = ManipulatorDynamics()
    integrator = NumericalIntegrators(robot.manipulator_ode)
    
    y0 = [0.1, 0.1, 0.0, 0.0]
    t_span = [0.0, 1.0]
    k = 100.0
    
    print("  Testando RK4...")
    t_rk4, y_rk4, stats_rk4 = integrator.rk4_explicit(t_span, y0, 0.01, k)
    
    print("  Testando Radau...")
    t_radau, y_radau, stats_radau = integrator.radau_scipy(t_span, y0, 1e-6, 1e-9, k)
    
    if stats_rk4.success and stats_radau.success:
        print(f"  ✓ RK4: {len(t_rk4)} punti, {stats_rk4.n_function_evaluations} eval f")
        print(f"  ✓ Radau: {len(t_radau)} punti, {stats_radau.n_function_evaluations} eval f")
        
        # Confronto finale
        error = abs(y_rk4[-1, 0] - y_radau[-1, 0])
        print(f"   Differenza finale θ1: {error:.2e} rad")
        
        return True
    else:
        print("  ✗ Errore nella demo")
        return False

def run_full_experiments():
    """Esegue la campagna completa di esperimenti."""
    print(" Avvio esperimenti completi...")
    
    # Stima del tempo richiesto
    print("  Stima tempo: 5-15 minuti (dipende dalla velocità del sistema)")
    proceed = input("  Procedere? (y/n): ").lower().strip()
    
    if proceed != 'y':
        print("  Esperimenti saltati.")
        return None
    
    start_time = time.time()
    
    # Esecuzione degli esperimenti
    experiment_manager = ExperimentManager()
    experiment_manager.run_experiments()
    
    # Salvataggio risultati
    df = experiment_manager.save_results("results_manipulator.csv")
    
    # Creazione grafici
    print(" Generazione grafici di analisi...")
    experiment_manager.create_analysis_plots(df)
    
    # Tabella riassuntiva
    experiment_manager.print_summary_table(df)
    
    total_time = time.time() - start_time
    print(f"\n Esperimenti completati in {total_time/60:.1f} minuti")
    
    return df

def run_trajectory_analysis():
    """Esegue l'analisi delle traiettorie e visualizzazioni."""
    print(" Analisi delle traiettorie...")
    
    viz = TrajectoryVisualizer()
    
    try:
        print("  → Traiettorie degli angoli dei giunti...")
        viz.plot_joint_trajectories(k_values=[10, 100, 500])
        
        print("  → Ritratti di fase...")
        viz.plot_phase_portraits(k_values=[10, 100, 500])
        
        print("  → Traiettorie end-effector...")
        viz.plot_end_effector_trajectory(k_values=[10, 100, 500])
        
        print("  → Evoluzione dell'energia...")
        viz.plot_energy_evolution(k_values=[10, 100, 500])
        
        print("  → Analisi di stabilità...")
        viz.compare_methods_stability(k=500)
        
        print("Analisi delle traiettorie completata")
        
        # Opzione per animazione
        create_anim = input("  Creare animazione del manipolatore? (y/n): ").lower().strip()
        if create_anim == 'y':
            print("  🎬 Creazione animazione...")
            anim = viz.animate_manipulator(k=100, method="Radau", rtol=1e-6, save_animation=True)
            print("   Animazione creata")
        
    except Exception as e:
        print(f"  ✗ Errore nell'analisi delle traiettorie: {e}")

def create_project_report():
    """Crea un report riassuntivo del progetto."""
    print(" Generazione report del progetto...")
    
    report_content = """
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
"""
    
    with open("project_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print("  Report salvato in 'project_report.md'")

def main():
    """Funzione principale che coordina l'esecuzione del progetto."""
    
    # Parse degli argomenti da linea di comando
    parser = argparse.ArgumentParser(description="Analisi Stabilità RK4 vs Radau")
    parser.add_argument("--quick", action="store_true", help="Esegui solo demo rapida")
    parser.add_argument("--no-experiments", action="store_true", help="Salta esperimenti completi")
    parser.add_argument("--no-plots", action="store_true", help="Salta generazione grafici")
    args = parser.parse_args()
    
    # Header del progetto
    print_header()
    
    # 1. Verifica installazione
    if not test_installation():
        print("Installazione non completa. Terminazione.")
        sys.exit(1)
    
    print()
    
    # 2. Demo rapida
    if not run_quick_demo():
        print("Demo rapida fallita. Verifica l'installazione.")
        sys.exit(1)
    
    print()
    
    # Se solo demo rapida, termina qui
    if args.quick:
        print("Demo rapida completata con successo!")
        return
    
    # 3. Esperimenti completi
    df_results = None
    if not args.no_experiments:
        df_results = run_full_experiments()
        print()
    
    # 4. Analisi delle traiettorie
    if not args.no_plots:
        run_trajectory_analysis()
        print()
    
    # 5. Creazione report
    create_project_report()
    
    # 6. Riepilogo finale
    print("\n" + "=" * 80)
    print("PROGETTO COMPLETATO CON SUCCESSO!")
    print("=" * 80)
    
    print("\nFile generati:")
    file_list = [
        "results_manipulator.csv",
        "error_vs_step.png",
        "efficiency_analysis.png", 
        "energy_conservation.png",
        "stability_regions.png",
        "joint_trajectories.png",
        "phase_portraits.png",
        "end_effector_trajectory.png",
        "energy_evolution.png",
        "stability_comparison.png",
        "project_report.md"
    ]
    
    for filename in file_list:
        if Path(filename).exists():
            print(f"  ✅ {filename}")
        else:
            print(f"  ❓ {filename} (non trovato)")
    
    if Path("manipulator_animation.gif").exists():
        print(f"  ✅ manipulator_animation.gif")
    
    print("\n Suggerimenti:")
    print("  - Apri i file .png per vedere i grafici")
    print("  - Leggi results_manipulator.csv per i dati numerici")
    print("  - Consulta project_report.md per il riepilogo")
    
    if df_results is not None and len(df_results) > 0:
        success_rate = len(df_results[df_results['success'] == True]) / len(df_results) * 100
        print(f"\n Statistiche: {success_rate:.1f}% di test completati con successo")


if __name__ == "__main__":
    main()