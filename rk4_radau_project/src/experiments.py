"""
experiments.py - Campagna di test per confrontare i metodi numerici

Esegue sweep parametrici sui passi h e tolleranze rtol per tutti i metodi,
raccogliendo metriche di accuratezza, stabilità e performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import time
from dynamics import ManipulatorDynamics
from integrators import NumericalIntegrators
import warnings
warnings.filterwarnings('ignore')

class ExperimentManager:
    """
    Gestisce gli esperimenti numerici per confrontare i metodi di integrazione.
    """
    
    def __init__(self):
        self.robot = ManipulatorDynamics()
        self.integrator = NumericalIntegrators(self.robot.manipulator_ode)
        
        # Configurazione dei test
        self.t_span = [0.0, 10.0]  # intervallo di simulazione [s]
        self.y0 = np.array([0.1, 0.1, 0.0, 0.0])  # condizioni iniziali
        
        # Valori di rigidezza da testare
        self.k_values = [10, 100, 500]  # [N⋅m/rad]
        
        # Griglie di test
        self.h_values = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]  # passi fissi
        self.rtol_values = [1e-3, 1e-4, 1e-5, 1e-6]  # tolleranze per Radau
        
        # Risultati
        self.results = []
        self.reference_solutions = {}
    
    def compute_reference_solution(self, k):
        """
        Calcola una soluzione di riferimento ad alta precisione per il calcolo degli errori.
        """
        print(f"Calcolando soluzione di riferimento per k = {k}...")
        
        try:
            # Usa Radau con tolleranza molto stretta
            # CORRECTED: k deve essere passato come argomento posizionale PRIMA dei keyword args
            t_ref, y_ref, stats = self.integrator.radau_scipy(
                self.t_span, 
                self.y0, 
                1e-10,    # rtol come posizionale
                1e-12,    # atol come posizionale  
                k         # k come posizionale (va negli *args)
            )
            
            if not stats.success:
                raise RuntimeError(f"Fallimento nel calcolo della soluzione di riferimento per k={k}: {stats.error_message}")
            
            self.reference_solutions[k] = (t_ref, y_ref)
            print(f"✓ Soluzione di riferimento calcolata: {len(t_ref)} punti")
            
        except Exception as e:
            raise RuntimeError(f"Errore nel calcolo della soluzione di riferimento per k={k}: {str(e)}")
    
    def compute_error_metrics(self, t_num, y_num, k):
        """
        Calcola le metriche di errore rispetto alla soluzione di riferimento.
        """
        if k not in self.reference_solutions:
            self.compute_reference_solution(k)
        
        t_ref, y_ref = self.reference_solutions[k]
        
        # Interpolazione della soluzione numerica sui tempi di riferimento
        y_interp = np.zeros_like(y_ref)
        
        for i in range(y_ref.shape[1]):  # per ogni componente dello stato
            y_interp[:, i] = np.interp(t_ref, t_num, y_num[:, i])
        
        # Errore RMS su q1 (prima componente angolare)
        error_q1 = np.sqrt(np.mean((y_interp[:, 0] - y_ref[:, 0])**2))
        
        # Errore RMS globale
        error_global = np.sqrt(np.mean(np.sum((y_interp - y_ref)**2, axis=1)))
        
        # Errore finale
        error_final = np.linalg.norm(y_interp[-1] - y_ref[-1])
        
        return {
            'error_q1_rms': error_q1,
            'error_global_rms': error_global,
            'error_final': error_final
        }
    
    def compute_energy_drift(self, t, y, k):
        """
        Calcola il drift energetico sul lungo periodo.
        """
        energies = np.array([self.robot.total_energy(state, k) for state in y])
        
        E0 = energies[0]
        E_final = energies[-1]
        
        # Drift relativo
        relative_drift = abs(E_final - E0) / abs(E0) if abs(E0) > 1e-12 else 0
        
        # Massima variazione
        max_variation = (np.max(energies) - np.min(energies)) / abs(E0) if abs(E0) > 1e-12 else 0
        
        return {
            'energy_initial': E0,
            'energy_final': E_final,
            'energy_drift_relative': relative_drift,
            'energy_variation_max': max_variation
        }
    
    def test_fixed_step_method(self, method_name, method_func, k, h):
        """
        Testa un metodo a passo fisso con parametri specifici.
        """
        try:
            start_time = time.time()
            t, y, stats = method_func(self.t_span, self.y0, h, k)
            total_time = time.time() - start_time
            
            result = {
                'method': method_name,
                'k': k,
                'h': h,
                'rtol': None,
                'success': stats.success,
                'n_steps': len(t) - 1,
                'nfev': stats.n_function_evaluations,
                'njev': stats.n_jacobian_evaluations,
                'newton_iter': stats.n_newton_iterations,
                'cpu_time': total_time,
                'error_message': stats.error_message
            }
            
            if stats.success and len(t) > 1:
                # Calcolo metriche di errore
                try:
                    error_metrics = self.compute_error_metrics(t, y, k)
                    result.update(error_metrics)
                    
                    # Calcolo drift energetico
                    energy_metrics = self.compute_energy_drift(t, y, k)
                    result.update(energy_metrics)
                    
                except Exception as e:
                    print(f"Errore nel calcolo delle metriche per {method_name}: {e}")
                    result['success'] = False
                    result['error_message'] = str(e)
            
            return result
            
        except Exception as e:
            return {
                'method': method_name,
                'k': k,
                'h': h,
                'rtol': None,
                'success': False,
                'error_message': str(e),
                'nfev': 0,
                'cpu_time': 0.0
            }
    
    def test_adaptive_method(self, method_name, method_func, k, rtol):
        """
        Testa un metodo adattivo (Radau) con tolleranza specifica.
        """
        try:
            start_time = time.time()
            # CORRECTED: Tutti gli argomenti posizionali prima dei keyword
            t, y, stats = method_func(self.t_span, self.y0, rtol, rtol*1e-3, k)
            total_time = time.time() - start_time
            
            result = {
                'method': method_name,
                'k': k,
                'h': None,
                'rtol': rtol,
                'success': stats.success,
                'n_steps': len(t) - 1,
                'nfev': stats.n_function_evaluations,
                'njev': stats.n_jacobian_evaluations,
                'newton_iter': 0,  # Radau gestisce internamente
                'cpu_time': total_time,
                'error_message': stats.error_message
            }
            
            if stats.success and len(t) > 1:
                try:
                    error_metrics = self.compute_error_metrics(t, y, k)
                    result.update(error_metrics)
                    
                    energy_metrics = self.compute_energy_drift(t, y, k)
                    result.update(energy_metrics)
                    
                except Exception as e:
                    print(f"Errore nel calcolo delle metriche per {method_name}: {e}")
                    result['success'] = False
                    result['error_message'] = str(e)
            
            return result
            
        except Exception as e:
            return {
                'method': method_name,
                'k': k,
                'h': None,
                'rtol': rtol,
                'success': False,
                'error_message': str(e),
                'nfev': 0,
                'cpu_time': 0.0
            }
    
    # def run_experiments(self):
    #     """
    #     Esegue la campagna completa di esperimenti.
    #     """
    #     print("=== Avvio campagna di esperimenti ===")
    #     print(f"Condizioni iniziali: {self.y0}")
    #     print(f"Intervallo temporale: {self.t_span}")
    #     print(f"Rigidezze: {self.k_values}")
        
    #     # Calcolo soluzioni di riferimento
    #     for k in self.k_values:
    #         self.compute_reference_solution(k)
        
    #     # Definizione dei metodi a passo fisso
    #     fixed_step_methods = [
    #         ("Euler_Explicit", self.integrator.euler_explicit),
    #         ("Euler_Implicit", self.integrator.euler_implicit),
    #         ("Trapezoid", self.integrator.trapezoid_implicit),
    #         ("RK4", self.integrator.rk4_explicit)
    #     ]
        
    #     total_tests = len(self.k_values) * (len(fixed_step_methods) * len(self.h_values) + len(self.rtol_values))
    #     current_test = 0
        
    #     print(f"Numero totale di test: {total_tests}")
        
    #     # Test dei metodi a passo fisso
    #     for k in self.k_values:
    #         print(f"\n--- Test con k = {k} N⋅m/rad ---")
            
    #         for method_name, method_func in fixed_step_methods:
    #             print(f"Testing {method_name}...")
                
    #             for h in self.h_values:
    #                 current_test += 1
    #                 print(f"  h = {h:.3f} ({current_test}/{total_tests})", end="")
                    
    #                 result = self.test_fixed_step_method(method_name, method_func, k, h)
    #                 self.results.append(result)
                    
    #                 if result['success']:
    #                     print(f" ✓ (nfev: {result['nfev']})")
    #                 else:
    #                     print(f" ✗ ({result['error_message']})")
            
    #         # Test Radau con diverse tolleranze
    #         print("Testing Radau...")
    #         for rtol in self.rtol_values:
    #             current_test += 1
    #             print(f"  rtol = {rtol:.0e} ({current_test}/{total_tests})", end="")
                
    #             # Chiamata corretta a Radau: t_span, y0, rtol, atol, *args
    #             result = self.test_adaptive_method("Radau", self.integrator.radau_scipy, k, rtol)
    #             self.results.append(result)
                
    #             if result['success']:
    #                 print(f" ✓ (nfev: {result['nfev']})")
    #             else:
    #                 print(f" ✗ ({result['error_message']})")
        
    #     print(f"\n✓ Esperimenti completati: {len(self.results)} test eseguiti")
    
    def run_experiments(self):
        """Esegue la campagna completa di esperimenti"""
        print("=== Avvio campagna di esperimenti ===")
        print(f"Condizioni iniziali: {self.y0}")
        print(f"Intervallo temporale: {self.t_span}")
        print(f"Rigidezze: {self.k_values}")
        
        # Calcolo soluzioni di riferimento
        for k in self.k_values:
            self.compute_reference_solution(k)
        
        # ESPERIMENTI ORIGINALI
        print("\n" + "="*50)
        print("ESPERIMENTI ORIGINALI")
        print("="*50)
        
        # ... (mantieni tutto il codice esistente) ...
        
        # NUOVO: CONFRONTO EQUO
        print("\n" + "="*50)
        print("CONFRONTO EQUO")
        print("="*50)
        
        fair_results = self.run_fair_comparison_experiments()
        
        # Aggiungi i risultati equi ai risultati principali
        self.results.extend(fair_results)
        
        total_results = len(self.results)
        print(f"\n✓ Esperimenti totali completati: {total_results} test")

    def save_results(self, filename="results.csv"):
        """
        Salva i risultati in formato CSV.
        """
        df = pd.DataFrame(self.results)
        df.to_csv(filename, index=False)
        print(f"Risultati salvati in {filename}")
        return df
    
    def create_analysis_plots(self, df=None):
        """
        Crea grafici di analisi dei risultati.
        """
        if df is None:
            df = pd.DataFrame(self.results)
        
        # Filtra solo i risultati di successo
        df_success = df[df['success'] == True].copy()
        
        if len(df_success) == 0:
            print("Nessun risultato di successo per creare i grafici!")
            return
        
        # Configura lo stile
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Errore RMS vs passo/tolleranza per ogni rigidezza
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, k in enumerate(self.k_values):
            ax = axes[i]
            
            # Dati per metodi a passo fisso
            df_fixed = df_success[(df_success['k'] == k) & (df_success['h'].notna())]
            
            for method in df_fixed['method'].unique():
                method_data = df_fixed[df_fixed['method'] == method]
                if len(method_data) > 0:
                    ax.loglog(method_data['h'], method_data['error_q1_rms'], 
                             'o-', label=method, markersize=6)
            
            # Dati per Radau (usa rtol come "passo equivalente")
            df_radau = df_success[(df_success['k'] == k) & (df_success['method'] == 'Radau')]
            if len(df_radau) > 0:
                ax.loglog(df_radau['rtol'], df_radau['error_q1_rms'], 
                         's-', label='Radau', markersize=8, linewidth=2)
            
            ax.set_xlabel('Passo h / Tolleranza rtol')
            ax.set_ylabel('Errore RMS su q₁')
            ax.set_title(f'k = {k} N⋅m/rad')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('error_vs_step.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Efficienza computazionale: errore vs tempo CPU
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, k in enumerate(self.k_values):
            ax = axes[i]
            
            df_k = df_success[df_success['k'] == k]
            
            for method in df_k['method'].unique():
                method_data = df_k[df_k['method'] == method]
                if len(method_data) > 0:
                    marker = 'o' if method != 'Radau' else 's'
                    ax.loglog(method_data['cpu_time'], method_data['error_q1_rms'], 
                             marker, label=method, markersize=6, alpha=0.7)
            
            ax.set_xlabel('Tempo CPU [s]')
            ax.set_ylabel('Errore RMS su q₁')
            ax.set_title(f'Efficienza - k = {k} N⋅m/rad')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('efficiency_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 3. Drift energetico
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, k in enumerate(self.k_values):
            ax = axes[i]
            
            df_k = df_success[df_success['k'] == k]
            
            for method in df_k['method'].unique():
                method_data = df_k[df_k['method'] == method]
                if len(method_data) > 0 and 'energy_drift_relative' in method_data.columns:
                    x_vals = method_data['h'].fillna(method_data['rtol'])
                    marker = 'o' if method != 'Radau' else 's'
                    ax.loglog(x_vals, method_data['energy_drift_relative'], 
                             marker, label=method, markersize=6, alpha=0.7)
            
            ax.set_xlabel('Passo h / Tolleranza rtol')
            ax.set_ylabel('Drift energetico relativo')
            ax.set_title(f'Conservazione energia - k = {k} N⋅m/rad')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.savefig('energy_conservation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Regioni di stabilità
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, k in enumerate(self.k_values):
            ax = axes[i]
            
            # Mappa di stabilità per metodi a passo fisso
            df_k = df[(df['k'] == k) & (df['h'].notna())]
            
            for method in ['Euler_Explicit', 'RK4', 'Euler_Implicit', 'Trapezoid']:
                method_data = df_k[df_k['method'] == method]
                if len(method_data) > 0:
                    successful = method_data[method_data['success'] == True]
                    failed = method_data[method_data['success'] == False]
                    
                    if len(successful) > 0:
                        ax.scatter(successful['h'], [method]*len(successful), 
                                 color='green', s=50, alpha=0.7, marker='o')
                    if len(failed) > 0:
                        ax.scatter(failed['h'], [method]*len(failed), 
                                 color='red', s=50, alpha=0.7, marker='x')
            
            ax.set_xscale('log')
            ax.set_xlabel('Passo h')
            ax.set_title(f'Regioni di stabilità - k = {k} N⋅m/rad')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('stability_regions.png', dpi=300, bbox_inches='tight')
        plt.show()
         # 5. NUOVO: Confronto equo sui passi di Radau
        print("  Creando grafico confronto equo...")
        self.plot_fair_comparison(df)


    

    def plot_fair_comparison(self, df):
        """Grafico con confronto equo sui passi di Radau"""
        print("  Creando grafico confronto equo...")
        
        # Filtra solo i risultati del confronto equo
        df_fair = df[df.get('comparison_type') == 'fair'].copy()
        df_success = df_fair[df_fair['success'] == True].copy()
        
        if len(df_success) == 0:
            print("    Nessun risultato di successo per confronto equo!")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        for i, k in enumerate(self.k_values):
            ax = axes[i]
            
            # Dati per questo k
            df_k = df_success[df_success['k'] == k]
            
            if len(df_k) == 0:
                continue
            
            # Radau (riferimento)
            radau_data = df_k[df_k['method'] == 'Radau']
            if len(radau_data) > 0:
                ax.loglog(radau_data['rtol'], radau_data['error_q1_rms'], 
                        'o-', label='Radau (adattivo)', linewidth=3, markersize=8)
            
            # Altri metodi sui passi di Radau
            colors = {'RK4': 'red', 'Euler_Explicit': 'orange', 'Euler_Implicit': 'brown', 'Trapezoid': 'green'}
            
            for method in ['RK4', 'Euler_Explicit', 'Euler_Implicit', 'Trapezoid']:
                method_data = df_k[df_k['method'] == method]
                
                if len(method_data) > 0:
                    color = colors.get(method, 'gray')
                    ax.loglog(method_data['radau_equivalent_rtol'], 
                            method_data['error_q1_rms'], 
                            's--', color=color, label=f'{method} (passo fisso)', 
                            alpha=0.7, markersize=6)
            
            ax.set_xlabel('Tolleranza Radau equivalente')
            ax.set_ylabel('Errore RMS su q₁')
            ax.set_title(f'Confronto equo - k = {k} N⋅m/rad')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fair_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def print_summary_table(self, df=None):
        """
        Stampa una tabella riassuntiva dei risultati migliori.
        """
        if df is None:
            df = pd.DataFrame(self.results)
        
        df_success = df[df['success'] == True].copy()
        
        print("\n=== TABELLA RIASSUNTIVA ===")
        print(f"{'k [N⋅m/rad]':<12} {'Metodo':<15} {'h/rtol':<10} {'Errore RMS':<12} {'CPU [ms]':<10} {'nfev':<6}")
        print("-" * 80)
        
        for k in self.k_values:
            df_k = df_success[df_success['k'] == k]
            
            if len(df_k) == 0:
                print(f"{k:<12} {'NESSUN SUCCESSO':<50}")
                continue
            
            # Trova il metodo più accurato
            best_accuracy = df_k.loc[df_k['error_q1_rms'].idxmin()]
            
            # Trova il metodo più efficiente (minor tempo per errore simile)
            df_efficient = df_k[df_k['error_q1_rms'] < 2 * best_accuracy['error_q1_rms']]
            if len(df_efficient) > 0:
                best_efficiency = df_efficient.loc[df_efficient['cpu_time'].idxmin()]
            else:
                best_efficiency = best_accuracy
            
            for label, row in [("Più accurato", best_accuracy), ("Più efficiente", best_efficiency)]:
                param = row['h'] if pd.notna(row['h']) else row['rtol']
                print(f"{k if label == 'Più accurato' else '':<12} "
                      f"{row['method']:<15} {param:<10.3e} {row['error_q1_rms']:<12.2e} "
                      f"{row['cpu_time']*1000:<10.1f} {row['nfev']:<6}")

    # Aggiungi dopo il metodo compute_energy_drift() in experiments.py

    def extract_radau_steps(self, k, rtol_values):
        """Estrae i passi effettivi usati da Radau per confronto equo"""
        print(f"  Estraendo passi Radau per k = {k}...")
        radau_steps = {}
        
        for rtol in rtol_values:
            t, y, stats = self.integrator.radau_scipy(
                self.t_span, self.y0, rtol, rtol*1e-3, k
            )
            
            if stats.success and len(t) > 1:
                # Calcola passi effettivi
                actual_steps = np.diff(t)
                avg_step = np.mean(actual_steps)
                min_step = np.min(actual_steps)
                max_step = np.max(actual_steps)
                
                radau_steps[rtol] = {
                    'avg_step': avg_step,
                    'min_step': min_step, 
                    'max_step': max_step,
                    'n_steps': len(t) - 1,
                    'rtol': rtol
                }
                print(f"    rtol={rtol:.0e}: passo medio = {avg_step:.6f}s")
        
        return radau_steps

    def run_fair_comparison_experiments(self):
        """Esegue confronto equo usando i passi di Radau"""
        print("\n=== CONFRONTO EQUO: Metodi con Passi di Radau ===")
        
        fair_results = []
        
        # Metodi a passo fisso da testare
        fixed_methods = [
            ("RK4", self.integrator.rk4_explicit),
            ("Euler_Explicit", self.integrator.euler_explicit),
            ("Euler_Implicit", self.integrator.euler_implicit),
            ("Trapezoid", self.integrator.trapezoid_implicit)
        ]
        
        for k in self.k_values:
            print(f"\n--- Confronto equo per k = {k} N⋅m/rad ---")
            
            # Estrai passi di Radau
            radau_steps = self.extract_radau_steps(k, self.rtol_values)
            
            # Test ogni metodo con i passi di Radau
            for rtol, step_info in radau_steps.items():
                h_radau = step_info['avg_step']
                
                print(f"  Testando con passo Radau (rtol={rtol:.0e}): h={h_radau:.6f}s")
                
                # Test di Radau stesso (per riferimento)
                radau_result = self.test_adaptive_method("Radau", self.integrator.radau_scipy, k, rtol)
                radau_result['h_equivalent'] = h_radau
                radau_result['comparison_type'] = 'fair'
                fair_results.append(radau_result)
                
                # Test altri metodi con stesso passo
                for method_name, method_func in fixed_methods:
                    result = self.test_fixed_step_method(method_name, method_func, k, h_radau)
                    result['radau_equivalent_rtol'] = rtol
                    result['comparison_type'] = 'fair'
                    fair_results.append(result)
                    
                    status = "✓" if result['success'] else "✗"
                    print(f"    {status} {method_name}")
        
        return fair_results
# Script principale
if __name__ == "__main__":
    # Creazione del manager degli esperimenti
    experiment = ExperimentManager()
    
    # Esecuzione degli esperimenti
    experiment.run_experiments()
    
    # Salvataggio e analisi
    df = experiment.save_results("manipulator_results.csv")
    experiment.create_analysis_plots(df)
    experiment.print_summary_table(df)
    
    print("\n✓ Analisi completa terminata!")
    print("File generati:")
    print("  - manipulator_results.csv")
    print("  - error_vs_step.png")
    print("  - efficiency_analysis.png") 
    print("  - energy_conservation.png")
    print("  - stability_regions.png")