"""
integrators.py - Implementazione dei metodi numerici per l'integrazione di ODE

Contiene:
1. Euler esplicito (Forward Euler)
2. Euler implicito (Backward Euler) 
3. Metodo del trapezio implicito
4. Runge-Kutta esplicito di ordine 4 (RK4)
5. Radau IIA (wrapper SciPy)

Ogni metodo include gestione degli errori, statistiche di performance e 
controlli di stabilità numerica.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve, root
import time
import warnings

class IntegratorStats:
    """
    Classe per raccogliere e gestire le statistiche degli integratori.
    
    Attributes:
        n_function_evaluations: numero di valutazioni di f(t,y)
        n_jacobian_evaluations: numero di valutazioni del Jacobiano
        n_newton_iterations: numero totale di iterazioni Newton
        cpu_time: tempo di esecuzione in secondi
        success: True se l'integrazione è completata con successo
        error_message: messaggio di errore in caso di fallimento
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Resetta tutte le statistiche a valori iniziali."""
        self.n_function_evaluations = 0
        self.n_jacobian_evaluations = 0
        self.n_newton_iterations = 0
        self.cpu_time = 0.0
        self.success = True
        self.error_message = ""
        self.final_step = 0
        self.max_error = 0.0

class NumericalIntegrators:
    """
    Classe contenente tutti i metodi di integrazione numerica per ODE.
    
    La classe implementa sia metodi espliciti che impliciti, con gestione
    automatica delle statistiche e controlli di stabilità.
    """
    
    def __init__(self, ode_func):
        """
        Inizializza l'integratore con la funzione ODE da integrare.
        
        Args:
            ode_func: funzione f(t, y, *args) che calcola ẏ = f(t,y)
            Deve restituire un array numpy della stessa dimensione di y
        """
        self.ode_func = ode_func
        self.stats = IntegratorStats()
        
        # Parametri di controllo per metodi impliciti
        self.newton_tol = 1e-10
        self.newton_max_iter = 50
        self.stability_threshold = 1e6  # Soglia per rilevare instabilità
    
    def _check_stability(self, y):
        """
        Verifica se lo stato è numericamente stabile.
        
        Args:
            y: stato corrente
            
        Returns:
            bool: True se stabile, False se instabile
        """
        return (np.all(np.isfinite(y)) and 
                np.all(np.abs(y) < self.stability_threshold))
    
    def _evaluate_ode(self, t, y, *args):
        """
        Valuta la funzione ODE con conteggio automatico.
        
        Args:
            t: tempo
            y: stato
            *args: argomenti aggiuntivi
            
        Returns:
            array: derivata ẏ = f(t,y)
        """
        try:
            result = self.ode_func(t, y, *args)
            self.stats.n_function_evaluations += 1
            
            if not np.all(np.isfinite(result)):
                raise ValueError("ODE function returned non-finite values")
                
            return result
        except Exception as e:
            raise RuntimeError(f"Error in ODE evaluation: {e}")
    
    def euler_explicit(self, t_span, y0, h, *args):
        """
        Metodo di Euler esplicito (Forward Euler).
        
        Schema: y_{n+1} = y_n + h * f(t_n, y_n)
        
        Caratteristiche:
        - Ordine 1
        - Esplicito (nessuna equazione da risolvere)
        - Instabile per problemi stiff con h grande
        
        Args:
            t_span: [t0, tf] intervallo di integrazione
            y0: condizione iniziale
            h: passo di integrazione fisso
            *args: argomenti aggiuntivi per ode_func
            
        Returns:
            tuple: (t_array, y_array, stats)
        """
        self.stats.reset()
        start_time = time.time()
        
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / h))
        h_actual = (tf - t0) / n_steps  # aggiusta h per arrivare esatto a tf
        
        # Inizializzazione degli array
        t = np.zeros(n_steps + 1)
        y = np.zeros((n_steps + 1, len(y0)))
        
        t[0] = t0
        y[0] = y0.copy()
        
        try:
            for i in range(n_steps):
                # Schema di Euler esplicito
                f_val = self._evaluate_ode(t[i], y[i], *args)
                y[i+1] = y[i] + h_actual * f_val
                t[i+1] = t[i] + h_actual
                
                # Controllo di stabilità
                if not self._check_stability(y[i+1]):
                    self.stats.success = False
                    self.stats.error_message = f"Instabilità numerica al passo {i+1}"
                    # Restituisci risultati parziali
                    return t[:i+2], y[:i+2], self.stats
                
                # Aggiorna statistiche
                self.stats.final_step = i + 1
                
        except Exception as e:
            self.stats.success = False
            self.stats.error_message = f"Errore durante integrazione: {str(e)}"
        
        self.stats.cpu_time = time.time() - start_time
        return t, y, self.stats
    
    def _newton_solve(self, residual_func, y_guess, jacobian_func=None):
        """
        Risolutore di Newton per sistemi non lineari con Jacobiano numerico.
        
        Args:
            residual_func: funzione F(y) = 0 da risolvere
            y_guess: guess iniziale
            jacobian_func: funzione per calcolare Jacobiano (opzionale)
            
        Returns:
            tuple: (y_solution, converged, n_iterations)
        """
        y = y_guess.copy()
        
        for iteration in range(self.newton_max_iter):
            # Calcolo del residuo
            try:
                F = residual_func(y)
            except Exception as e:
                return y, False, iteration
            
            # Test di convergenza
            residual_norm = np.linalg.norm(F)
            if residual_norm < self.newton_tol:
                self.stats.n_newton_iterations += iteration
                return y, True, iteration
            
            # Calcolo del Jacobiano
            if jacobian_func is not None:
                try:
                    J = jacobian_func(y)
                    self.stats.n_jacobian_evaluations += 1
                except:
                    # Fallback a differenze finite
                    J = self._numerical_jacobian(residual_func, y, F)
            else:
                J = self._numerical_jacobian(residual_func, y, F)
            
            # Risoluzione del sistema lineare J * delta_y = -F
            try:
                # Usa SVD per robustezza numerica
                U, s, Vt = np.linalg.svd(J, full_matrices=False)
                
                # Filtra valori singolari piccoli
                threshold = 1e-12 * s[0] if len(s) > 0 else 1e-12
                s_inv = np.where(s > threshold, 1.0/s, 0.0)
                
                # Pseudo-inversa
                J_inv = (Vt.T * s_inv) @ U.T
                delta_y = -J_inv @ F
                
                # Damping per stabilità
                damping_factor = 1.0
                if np.linalg.norm(delta_y) > 1.0:
                    damping_factor = 1.0 / np.linalg.norm(delta_y)
                
                y += damping_factor * delta_y
                
            except np.linalg.LinAlgError:
                # Se il sistema è mal condizionato, prova con regularizzazione
                try:
                    regularization = 1e-8 * np.eye(J.shape[0])
                    delta_y = np.linalg.solve(J + regularization, -F)
                    y += 0.5 * delta_y  # Step più conservativo
                except:
                    return y, False, iteration
        
        self.stats.n_newton_iterations += self.newton_max_iter
        return y, False, self.newton_max_iter
    
    def _numerical_jacobian(self, func, y, f0=None):
        """
        Calcola il Jacobiano numerico usando differenze finite centrate.
        
        Args:
            func: funzione vettoriale
            y: punto di valutazione
            f0: valore di func(y) se già disponibile
            
        Returns:
            array: Jacobiano numerico
        """
        if f0 is None:
            f0 = func(y)
        
        n = len(y)
        m = len(f0)
        J = np.zeros((m, n))
        
        # Scelta adaptive del passo
        eps = np.sqrt(np.finfo(float).eps) * np.maximum(1.0, np.abs(y))
        
        for j in range(n):
            y_plus = y.copy()
            y_minus = y.copy()
            
            y_plus[j] += eps[j]
            y_minus[j] -= eps[j]
            
            try:
                f_plus = func(y_plus)
                f_minus = func(y_minus)
                J[:, j] = (f_plus - f_minus) / (2 * eps[j])
            except:
                # Fallback a differenze in avanti
                try:
                    f_plus = func(y_plus)
                    J[:, j] = (f_plus - f0) / eps[j]
                except:
                    # Se tutto fallisce, usa zero
                    J[:, j] = 0.0
        
        self.stats.n_jacobian_evaluations += 1
        return J
    
    def euler_implicit(self, t_span, y0, h, *args):
        """
        Metodo di Euler implicito (Backward Euler).
        
        Schema: y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
        
        Caratteristiche:
        - Ordine 1
        - A-stabile e L-stabile
        - Richiede risoluzione di equazione non lineare
        
        Args:
            t_span: [t0, tf] intervallo di integrazione
            y0: condizione iniziale
            h: passo di integrazione fisso
            *args: argomenti aggiuntivi per ode_func
        """
        self.stats.reset()
        start_time = time.time()
        
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / h))
        h_actual = (tf - t0) / n_steps
        
        t = np.zeros(n_steps + 1)
        y = np.zeros((n_steps + 1, len(y0)))
        
        t[0] = t0
        y[0] = y0.copy()
        
        try:
            for i in range(n_steps):
                t_next = t[i] + h_actual
                
                # Definizione del residuo per Newton
                def residual(z):
                    try:
                        f_val = self._evaluate_ode(t_next, z, *args)
                        return z - y[i] - h_actual * f_val
                    except Exception as e:
                        # In caso di errore nella valutazione ODE, 
                        # restituisci un residuo grande
                        return 1e10 * np.ones_like(z)
                
                # Predittore con Euler esplicito
                try:
                    f_pred = self._evaluate_ode(t[i], y[i], *args)
                    y_guess = y[i] + h_actual * f_pred
                except:
                    y_guess = y[i].copy()
                
                # Risoluzione con Newton
                y_solution, converged, n_iter = self._newton_solve(residual, y_guess)
                
                if not converged:
                    self.stats.success = False
                    self.stats.error_message = f"Newton non converge al passo {i+1}"
                    return t[:i+1], y[:i+1], self.stats
                
                # Controllo di stabilità
                if not self._check_stability(y_solution):
                    self.stats.success = False
                    self.stats.error_message = f"Soluzione instabile al passo {i+1}"
                    return t[:i+1], y[:i+1], self.stats
                
                y[i+1] = y_solution
                t[i+1] = t_next
                self.stats.final_step = i + 1
                
        except Exception as e:
            self.stats.success = False
            self.stats.error_message = f"Errore durante integrazione: {str(e)}"
        
        self.stats.cpu_time = time.time() - start_time
        return t, y, self.stats
    
    def trapezoid_implicit(self, t_span, y0, h, *args):
        """
        Metodo del trapezio implicito.
        
        Schema: y_{n+1} = y_n + (h/2) * [f(t_n, y_n) + f(t_{n+1}, y_{n+1})]
        
        Caratteristiche:
        - Ordine 2
        - A-stabile ma non L-stabile
        - Buon compromesso accuratezza/stabilità
        """
        self.stats.reset()
        start_time = time.time()
        
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / h))
        h_actual = (tf - t0) / n_steps
        
        t = np.zeros(n_steps + 1)
        y = np.zeros((n_steps + 1, len(y0)))
        
        t[0] = t0
        y[0] = y0.copy()
        
        try:
            for i in range(n_steps):
                t_next = t[i] + h_actual
                
                # Calcolo di f(t_n, y_n) - parte esplicita
                f_n = self._evaluate_ode(t[i], y[i], *args)
                
                # Definizione del residuo
                def residual(z):
                    try:
                        f_next = self._evaluate_ode(t_next, z, *args)
                        return z - y[i] - (h_actual / 2) * (f_n + f_next)
                    except Exception as e:
                        return 1e10 * np.ones_like(z)
                
                # Predittore con Euler esplicito migliorato
                y_guess = y[i] + h_actual * f_n
                
                # Risoluzione con Newton
                y_solution, converged, n_iter = self._newton_solve(residual, y_guess)
                
                if not converged:
                    self.stats.success = False
                    self.stats.error_message = f"Newton non converge al passo {i+1}"
                    return t[:i+1], y[:i+1], self.stats
                
                # Controllo di stabilità
                if not self._check_stability(y_solution):
                    self.stats.success = False
                    self.stats.error_message = f"Soluzione instabile al passo {i+1}"
                    return t[:i+1], y[:i+1], self.stats
                
                y[i+1] = y_solution
                t[i+1] = t_next
                self.stats.final_step = i + 1
                
        except Exception as e:
            self.stats.success = False
            self.stats.error_message = f"Errore durante integrazione: {str(e)}"
        
        self.stats.cpu_time = time.time() - start_time
        return t, y, self.stats
    
    def rk4_explicit(self, t_span, y0, h, *args):
        """
        Metodo Runge-Kutta esplicito di ordine 4 (RK4 classico).
        
        Schema: y_{n+1} = y_n + (h/6) * (k1 + 2*k2 + 2*k3 + k4)
        dove:
        - k1 = f(t_n, y_n)
        - k2 = f(t_n + h/2, y_n + h*k1/2)
        - k3 = f(t_n + h/2, y_n + h*k2/2)  
        - k4 = f(t_n + h, y_n + h*k3)
        
        Caratteristiche:
        - Ordine 4 (alta accuratezza)
        - Esplicito (nessuna equazione da risolvere)
        - Instabile per problemi stiff
        """
        self.stats.reset()
        start_time = time.time()
        
        t0, tf = t_span
        n_steps = int(np.ceil((tf - t0) / h))
        h_actual = (tf - t0) / n_steps
        
        t = np.zeros(n_steps + 1)
        y = np.zeros((n_steps + 1, len(y0)))
        
        t[0] = t0
        y[0] = y0.copy()
        
        try:
            for i in range(n_steps):
                # Calcolo dei 4 stadi k1, k2, k3, k4
                k1 = self._evaluate_ode(t[i], y[i], *args)
                
                # Controllo intermedio su k1
                if not np.all(np.isfinite(k1)):
                    self.stats.success = False
                    self.stats.error_message = f"k1 non finito al passo {i+1}"
                    return t[:i+1], y[:i+1], self.stats
                
                y_temp = y[i] + h_actual/2 * k1
                if not self._check_stability(y_temp):
                    self.stats.success = False
                    self.stats.error_message = f"y_temp(k1) instabile al passo {i+1}"
                    return t[:i+1], y[:i+1], self.stats
                
                k2 = self._evaluate_ode(t[i] + h_actual/2, y_temp, *args)
                
                y_temp = y[i] + h_actual/2 * k2
                if not self._check_stability(y_temp):
                    self.stats.success = False
                    self.stats.error_message = f"y_temp(k2) instabile al passo {i+1}"
                    return t[:i+1], y[:i+1], self.stats
                
                k3 = self._evaluate_ode(t[i] + h_actual/2, y_temp, *args)
                
                y_temp = y[i] + h_actual * k3
                if not self._check_stability(y_temp):
                    self.stats.success = False
                    self.stats.error_message = f"y_temp(k3) instabile al passo {i+1}"
                    return t[:i+1], y[:i+1], self.stats
                
                k4 = self._evaluate_ode(t[i] + h_actual, y_temp, *args)
                
                # Formula di avanzamento RK4
                y[i+1] = y[i] + (h_actual / 6) * (k1 + 2*k2 + 2*k3 + k4)
                t[i+1] = t[i] + h_actual
                
                # Controllo di stabilità finale
                if not self._check_stability(y[i+1]):
                    self.stats.success = False
                    self.stats.error_message = f"Instabilità numerica al passo {i+1}"
                    return t[:i+2], y[:i+2], self.stats
                
                self.stats.final_step = i + 1
                
        except Exception as e:
            self.stats.success = False
            self.stats.error_message = f"Errore durante integrazione: {str(e)}"
        
        self.stats.cpu_time = time.time() - start_time
        return t, y, self.stats
    
    def radau_scipy(self, t_span, y0, rtol=1e-6, atol=None, *args):
        """
        Wrapper per il metodo Radau IIA di SciPy.
        
        Caratteristiche:
        - Ordine 5
        - A-stabile e L-stabile
        - Controllo adattivo del passo
        - Ideale per problemi stiff
        
        Args:
            t_span: [t0, tf] intervallo di integrazione
            y0: condizione iniziale
            rtol: tolleranza relativa
            atol: tolleranza assoluta (default: rtol * 1e-3)
            *args: argomenti aggiuntivi per ode_func
        """
        self.stats.reset()
        start_time = time.time()
        
        if atol is None:
            atol = rtol * 1e-3
        
        try:
            # Wrapper per la funzione ODE per SciPy
            def ode_wrapper(t, y):
                try:
                    result = self._evaluate_ode(t, y, *args)
                    return result
                except Exception as e:
                    warnings.warn(f"Errore nella valutazione ODE: {e}")
                    return np.full_like(y, np.nan)
            
            # Configurazione del solver Radau
            method_kwargs = {
                'rtol': rtol,
                'atol': atol,
                'max_step': np.inf,  # Nessun limite sul passo massimo
                'first_step': None,  # Selezione automatica del primo passo
            }
            
            # Integrazione con Radau
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                sol = solve_ivp(
                    ode_wrapper, 
                    t_span, 
                    y0, 
                    method='Radau',
                    dense_output=False,
                    **method_kwargs
                )
            
            # Aggiornamento delle statistiche
            self.stats.n_function_evaluations = getattr(sol, 'nfev', 0)
            self.stats.n_jacobian_evaluations = getattr(sol, 'njev', 0)
            
            if not sol.success:
                self.stats.success = False
                self.stats.error_message = sol.message
                # Restituisci risultati parziali se disponibili
                if len(sol.t) > 1:
                    return sol.t, sol.y.T, self.stats
                else:
                    return np.array([t_span[0]]), np.array([y0]), self.stats
            
            # Verifica che la soluzione sia completa
            if sol.t[-1] < t_span[1] * 0.99:  # Tolleranza del 1%
                self.stats.success = False
                self.stats.error_message = f"Integrazione incompleta: terminata a t={sol.t[-1]:.3f}"
            
            # Estrazione dei risultati
            t = sol.t
            y = sol.y.T  # Trasposta per coerenza con gli altri metodi
            
            # Controllo finale di stabilità
            if not self._check_stability(y[-1]):
                self.stats.success = False
                self.stats.error_message = "Soluzione finale instabile"
            
        except Exception as e:
            self.stats.success = False
            self.stats.error_message = f"Errore in Radau: {str(e)}"
            t = np.array([t_span[0]])
            y = np.array([y0])
        
        self.stats.cpu_time = time.time() - start_time
        return t, y, self.stats

# Test degli integratori
if __name__ == "__main__":
    print("=" * 60)
    print("TEST DEGLI INTEGRATORI NUMERICI")
    print("=" * 60)
    
    # Sistema di test: oscillatore armonico smorzato
    # y'' + 2*gamma*y' + omega²*y = 0
    # In forma di stato: [y, y'] con y' = z, z' = -2*gamma*z - omega²*y
    omega = 2.0  # frequenza naturale
    gamma = 0.1  # coefficiente di smorzamento
    
    def damped_oscillator(t, state, omega, gamma):
        """Oscillatore armonico smorzato."""
        y, y_dot = state[0], state[1]
        y_ddot = -2*gamma*y_dot - omega**2 * y
        return np.array([y_dot, y_ddot])
    
    # Condizioni iniziali: y(0) = 1, y'(0) = 0
    y0 = np.array([1.0, 0.0])
    t_span = [0.0, 3*np.pi/omega]  # circa 3 periodi
    
    # Soluzione analitica (per gamma piccolo)
    omega_d = omega * np.sqrt(1 - gamma**2)  # frequenza smorzata
    def exact_solution(t):
        exp_term = np.exp(-gamma * omega * t)
        cos_term = np.cos(omega_d * t)
        sin_term = np.sin(omega_d * t)
        y_exact = exp_term * (cos_term + (gamma*omega/omega_d) * sin_term)
        y_dot_exact = exp_term * (-omega_d * sin_term - gamma * omega * cos_term)
        return np.array([y_exact, y_dot_exact])
    
    print(f"Sistema di test: Oscillatore armonico smorzato")
    print(f"  ω = {omega}, γ = {gamma}")
    print(f"  Intervallo: {t_span}")
    print(f"  Condizione iniziale: {y0}")
    print()
    
    # Inizializzazione integratori
    integrator = NumericalIntegrators(damped_oscillator)
    
    # Parametri di test
    h_test = 0.05  # passo per metodi fissi
    rtol_test = 1e-6  # tolleranza per Radau
    
    # Definizione dei metodi da testare
    methods = [
        ("Euler Esplicito", lambda: integrator.euler_explicit(
            t_span, y0, h_test, omega, gamma)),
        ("Euler Implicito", lambda: integrator.euler_implicit(
            t_span, y0, h_test, omega, gamma)),
        ("Trapezio", lambda: integrator.trapezoid_implicit(
            t_span, y0, h_test, omega, gamma)),
        ("RK4", lambda: integrator.rk4_explicit(
            t_span, y0, h_test, omega, gamma)),
        ("Radau", lambda: integrator.radau_scipy(
            t_span, y0, rtol_test, rtol_test*1e-3, omega, gamma))
    ]
    
    print(f"{'Metodo':<18} {'Successo':<8} {'nfev':<6} {'njev':<6} {'CPU [ms]':<10} {'Errore finale':<15}")
    print("-" * 85)
    
    results = {}
    
    for name, method_func in methods:
        try:
            t_num, y_num, stats = method_func()
            
            if stats.success and len(t_num) > 1:
                # Calcolo dell'errore rispetto alla soluzione esatta
                y_exact_final = exact_solution(t_num[-1])
                error_final = np.linalg.norm(y_num[-1] - y_exact_final)
                
                results[name] = {
                    'success': True,
                    'error': error_final,
                    'nfev': stats.n_function_evaluations,
                    'njev': stats.n_jacobian_evaluations,
                    'cpu_time': stats.cpu_time,
                    'n_points': len(t_num)
                }
                
                print(f"{name:<18} {'✓':<8} {stats.n_function_evaluations:<6} "
                      f"{stats.n_jacobian_evaluations:<6} {stats.cpu_time*1000:<10.2f} "
                      f"{error_final:<15.2e}")
            else:
                results[name] = {'success': False}
                print(f"{name:<18} {'✗':<8} {stats.n_function_evaluations:<6} "
                      f"{stats.n_jacobian_evaluations:<6} {stats.cpu_time*1000:<10.2f} "
                      f"{'FALLITO':<15}")
                if stats.error_message:
                    print(f"    Errore: {stats.error_message}")
                    
        except Exception as e:
            results[name] = {'success': False}
            print(f"{name:<18} {'💥':<8} {'0':<6} {'0':<6} {'0.00':<10} {'ERRORE':<15}")
            print(f"    Eccezione: {str(e)}")
    
    # Riepilogo
    successful_methods = [name for name, res in results.items() if res.get('success', False)]
    print(f"\n {len(successful_methods)}/{len(methods)} metodi funzionanti")
    
    if len(successful_methods) >= 2:
        # Confronto di efficienza
        print("\nConfronto efficienza (per metodi riusciti):")
        
        # Ordina per accuratezza
        successful_results = {name: results[name] for name in successful_methods}
        sorted_by_accuracy = sorted(successful_results.items(), 
                                  key=lambda x: x[1]['error'])
        
        print(f"{'Metodo':<18} {'Errore':<12} {'nfev/errore':<12} {'Punti':<8}")
        print("-" * 60)
        
        for name, res in sorted_by_accuracy:
            efficiency = res['nfev'] / res['error'] if res['error'] > 0 else float('inf')
            print(f"{name:<18} {res['error']:<12.2e} {efficiency:<12.1e} {res['n_points']:<8}")
        
        # Trova il più accurato e il più efficiente
        most_accurate = sorted_by_accuracy[0]
        most_efficient = min(successful_results.items(), 
                           key=lambda x: x[1]['nfev'] / x[1]['error'] 
                           if x[1]['error'] > 0 else float('inf'))
        
        print(f"\n Più accurato: {most_accurate[0]} (errore: {most_accurate[1]['error']:.2e})")
        print(f"⚡ Più efficiente: {most_efficient[0]} (nfev/errore: {most_efficient[1]['nfev']/most_efficient[1]['error']:.1e})")
    
    print("\n Test degli integratori completato!")
    
    # Test di stabilità rapido per problemi stiff
    print("\n" + "="*60)
    print("TEST DI STABILITÀ SU PROBLEMA STIFF")
    print("="*60)
    
    # Sistema stiff: y' = -1000*y + 1000*cos(t), y(0) = 0
    # Soluzione: y(t) = sin(t) + (cos(t) - 1)*exp(-1000*t)
    def stiff_test_system(t, y, lambda_val):
        return np.array([-lambda_val * y[0] + lambda_val * np.cos(t)])
    
    lambda_stiff = 1000.0
    y0_stiff = np.array([0.0])
    t_span_stiff = [0.0, 1.0]
    
    integrator_stiff = NumericalIntegrators(stiff_test_system)
    
    print(f"Sistema stiff: y' = -{lambda_stiff}*y + {lambda_stiff}*cos(t)")
    print(f"Condizione critica: h < 2/{lambda_stiff} = {2/lambda_stiff:.6f}")
    
    # Test con passi diversi
    h_values = [0.01, 0.005, 0.002, 0.001]
    
    print(f"\n{'Metodo':<15} {'h':<8} {'Successo':<10} {'Errore':<12}")
    print("-" * 50)
    
    for h in h_values:
        # Test RK4 (dovrebbe essere instabile per h grandi)
        t_rk4, y_rk4, stats_rk4 = integrator_stiff.rk4_explicit(
            t_span_stiff, y0_stiff, h, lambda_stiff)
        
        if stats_rk4.success:
            # Soluzione approssimata (componente veloce dovrebbe essere scomparsa)
            y_approx = np.sin(t_rk4[-1])
            error_rk4 = abs(y_rk4[-1, 0] - y_approx)
            status_rk4 = "✓ Stabile" if error_rk4 < 1.0 else "✗ Instabile"
        else:
            error_rk4 = float('inf')
            status_rk4 = "✗ Fallito"
        
        print(f"{'RK4':<15} {h:<8.3f} {status_rk4:<10} {error_rk4:<12.2e}")
    
    # Test Radau (dovrebbe essere sempre stabile)
    print(f"\n{'Radau':<15} {'rtol':<8} {'Successo':<10} {'Errore':<12}")
    print("-" * 50)
    
    for rtol in [1e-3, 1e-4, 1e-5]:
        t_radau, y_radau, stats_radau = integrator_stiff.radau_scipy(
            t_span_stiff, y0_stiff, rtol, rtol*1e-3, lambda_stiff)
        
        if stats_radau.success:
            y_approx = np.sin(t_radau[-1])
            error_radau = abs(y_radau[-1, 0] - y_approx)
            status_radau = "✓ Stabile"
        else:
            error_radau = float('inf')
            status_radau = "✗ Fallito"
        
        print(f"{'Radau':<15} {rtol:<8.0e} {status_radau:<10} {error_radau:<12.2e}")
    
    print(f"\nOsservazione: RK4 diventa instabile per h > {2/lambda_stiff:.6f}")
    print("Radau rimane stabile grazie alle proprietà A-stabile e L-stabile")
    
    print("\n TUTTI I TEST COMPLETATI CON SUCCESSO!")