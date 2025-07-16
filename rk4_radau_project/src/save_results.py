"""
save_results.py - Script per organizzare e salvare tutti i risultati del progetto

Questo script:
1. Crea una struttura di cartelle organizzata
2. Genera tutti i grafici mancanti
3. Salva tutto in cartelle dedicate
4. Crea un report riassuntivo
"""

import os
import glob
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def create_directory_structure():
    """Crea la struttura di cartelle per i risultati."""
    print("Creazione struttura cartelle...")
    
    directories = [
        "results",
        "results/graphs", 
        "results/data",
        "results/animations",
        "results/reports"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {dir_path}")

def move_existing_files():
    """Sposta i file esistenti nelle cartelle appropriate."""
    print("\n Spostamento file esistenti...")
    
    # Sposta PNG esistenti
    png_files = glob.glob("*.png")
    for file in png_files:
        dest = f"results/graphs/{file}"
        shutil.move(file, dest)
        print(f"  ✓ {file} → {dest}")
    
    # Sposta CSV esistenti
    csv_files = glob.glob("*.csv")
    for file in csv_files:
        dest = f"results/data/{file}"
        shutil.move(file, dest)
        print(f"  ✓ {file} → {dest}")
    
    # Sposta GIF esistenti
    gif_files = glob.glob("*.gif")
    for file in gif_files:
        dest = f"results/animations/{file}"
        shutil.move(file, dest)
        print(f"  ✓ {file} → {dest}")

def generate_experiment_graphs():
    """Genera i grafici degli esperimenti se mancanti."""
    print("\n Generazione grafici esperimenti...")
    
    try:
        from experiments import ExperimentManager
        
        # Controlla se abbiamo i dati
        csv_path = "results/data/manipulator_results.csv"
        if not os.path.exists(csv_path):
            print("  File manipulator_results.csv non trovato!")
            return False
        
        # Carica i dati
        df = pd.read_csv(csv_path)
        print(f"  ✓ Caricati {len(df)} risultati")
        
        # Crea i grafici
        exp = ExperimentManager()
        exp.results = df.to_dict('records')  # Converte per compatibilità
        
        # Disabilita display interattivo
        plt.ioff()
        
        # Crea i grafici nella directory corrente
        exp.create_analysis_plots(df)
        
        # Sposta i nuovi grafici
        new_graphs = glob.glob("*.png")
        for graph in new_graphs:
            dest = f"results/graphs/{graph}"
            shutil.move(graph, dest)
            print(f"  ✓ {graph} → {dest}")
        
        return True
        
    except Exception as e:
        print(f"  Errore nella generazione grafici esperimenti: {e}")
        return False

def generate_trajectory_graphs():
    """Genera i grafici delle traiettorie."""
    print("\n Generazione grafici traiettorie...")
    
    try:
        from trajectory_visualization import TrajectoryVisualizer
        
        viz = TrajectoryVisualizer()
        
        # Disabilita display interattivo
        plt.ioff()
        
        # Lista dei grafici da generare
        graph_functions = [
            ("Traiettorie giunti", lambda: viz.plot_joint_trajectories(k_values=[10, 100, 500], save_fig=True)),
            ("Ritratti di fase", lambda: viz.plot_phase_portraits(k_values=[10, 100, 500], save_fig=True)),
            ("Traiettorie end-effector", lambda: viz.plot_end_effector_trajectory(k_values=[10, 100, 500], save_fig=True)),
            ("Evoluzione energia", lambda: viz.plot_energy_evolution(k_values=[10, 100, 500], save_fig=True)),
            ("Confronto stabilità", lambda: viz.compare_methods_stability(k=500, save_fig=True))
        ]
        
        for name, func in graph_functions:
            try:
                print(f"  → {name}...")
                func()
                
                # Sposta eventuali nuovi grafici
                new_graphs = glob.glob("*.png")
                for graph in new_graphs:
                    dest = f"results/graphs/{graph}"
                    if not os.path.exists(dest):  # Evita sovrascritture
                        shutil.move(graph, dest)
                        print(f"    ✓ {graph} salvato")
                    else:
                        os.remove(graph)  # Rimuovi duplicato
                
            except Exception as e:
                print(f"     Errore in {name}: {e}")
        
        return True
        
    except Exception as e:
        print(f"   Errore generale grafici traiettorie: {e}")
        return False

def create_summary_report():
    """Crea un report riassuntivo del progetto."""
    print("\n Creazione report riassuntivo...")
    
    try:
        # Conta i file generati
        graphs = list(Path("results/graphs").glob("*.png"))
        data_files = list(Path("results/data").glob("*.csv"))
        animations = list(Path("results/animations").glob("*.gif"))
        
        report_content = f"""# REPORT PROGETTO: Stabilità RK4 vs Radau

##  Risultati Generati

### Grafici ({len(graphs)} file)
"""
        
        for graph in sorted(graphs):
            report_content += f"- `{graph.name}`\n"
        
        report_content += f"""
### Dati ({len(data_files)} file)
"""
        
        for data in sorted(data_files):
            report_content += f"- `{data.name}`\n"
        
        if animations:
            report_content += f"""
### Animazioni ({len(animations)} file)
"""
            for anim in sorted(animations):
                report_content += f"- `{anim.name}`\n"
        
        report_content += """
## Struttura dei Risultati

```
results/
├── graphs/          # Grafici di analisi e traiettorie
├── data/           # Dati numerici CSV
├── animations/     # Animazioni GIF (opzionali)
└── reports/        # Questo report
```

## Metodi Analizzati

1. **Euler Esplicito** (ordine 1, esplicito)
2. **Euler Implicito** (ordine 1, A-stabile, L-stabile)
3. **Metodo del Trapezio** (ordine 2, A-stabile)
4. **Runge-Kutta 4** (ordine 4, esplicito) - PROTAGONISTA
5. **Radau IIA** (ordine 5, A-stabile, L-stabile) - PROTAGONISTA

## 🎓 Conclusioni Attese

- **RK4**: Ottima accuratezza per problemi non-stiff, instabile per k grandi
- **Radau**: Sempre stabile, ideale per problemi stiff
- **Trade-off**: accuratezza vs stabilità vs costo computazionale

##  Files per la Relazione

Tutti i grafici sono pronti per essere inclusi nella relazione finale.
I dati numerici sono disponibili in formato CSV per analisi aggiuntive.

---
Generato automaticamente da save_results.py
"""
        
        # Salva il report
        report_path = "results/reports/project_summary.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)
        
        print(f"  ✓ Report salvato in {report_path}")
        return True
        
    except Exception as e:
        print(f"  ❌ Errore nella creazione report: {e}")
        return False

def print_final_summary():
    """Stampa il riepilogo finale."""
    print("\n" + "="*60)
    print("🎉 SALVATAGGIO RISULTATI COMPLETATO!")
    print("="*60)
    
    # Conta i file
    graphs = list(Path("results/graphs").glob("*.png"))
    data_files = list(Path("results/data").glob("*.csv"))
    animations = list(Path("results/animations").glob("*.gif"))
    
    print(f"\n📊 Risultati salvati:")
    print(f"  📈 Grafici: {len(graphs)} file in results/graphs/")
    print(f"  📄 Dati: {len(data_files)} file in results/data/")
    print(f"  🎬 Animazioni: {len(animations)} file in results/animations/")
    
    print(f"\n📁 Struttura creata:")
    for root, dirs, files in os.walk("results"):
        level = root.replace("results", "").count(os.sep)
        indent = " " * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = " " * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    print(f"\n💡 Per la relazione:")
    print(f"  - Tutti i grafici sono in results/graphs/")
    print(f"  - I dati numerici sono in results/data/")
    print(f"  - Il report è in results/reports/project_summary.md")

def main():
    """Funzione principale del salvataggio."""
    print("🔧 SALVATAGGIO E ORGANIZZAZIONE RISULTATI PROGETTO")
    print("="*60)
    
    # 1. Crea struttura cartelle
    create_directory_structure()
    
    # 2. Sposta file esistenti
    move_existing_files()
    
    # 3. Genera grafici esperimenti
    exp_success = generate_experiment_graphs()
    
    # 4. Genera grafici traiettorie
    traj_success = generate_trajectory_graphs()
    
    # 5. Crea report
    report_success = create_summary_report()
    
    # 6. Riepilogo finale
    print_final_summary()
    
    # Status finale
    if exp_success and traj_success and report_success:
        print("\n✅ TUTTI I RISULTATI SALVATI CON SUCCESSO!")
    else:
        print("\n⚠️  Alcuni problemi durante il salvataggio, ma i file principali sono stati creati.")
    
    print("\n🎓 Pronto per scrivere la relazione!")

if __name__ == "__main__":
    main()