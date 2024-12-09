from flask import Flask, render_template, request
import numpy as np
import matplotlib.pyplot as plt
import os
import time

app = Flask(__name__)

# Ensure the static folder exists
STATIC_FOLDER = "static"
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# ---------- Utility Functions ----------
def save_plot(filename):
    """
    Save the current plot to the static folder and close the figure.
    """
    filepath = os.path.join(STATIC_FOLDER, filename)
    plt.savefig(filepath)
    plt.close()
    return filepath

# ---------- Functions for App 1 ----------
def generate_plots(Vg, temp):
    q = 1.602e-19
    kb = 1.381e-23
    c1, c2, cg = 1.0e-20, 2.1e-19, 1.0e-18
    ctotal = c1 + c2 + cg
    r1, r2 = 15 * 1_000_000, 250 * 1_000_000  # Resistances in ohms
    q0 = 0
    vmin, vmax = -0.5, 0.5
    NV = 1000
    Vd = np.linspace(vmin, vmax, NV)
    Nmin, Nmax = -20, 20
    I = np.zeros(NV)

    for iv in range(NV):
        T1p, T1n, T2p, T2n = [], [], [], []
        for n in range(Nmin, Nmax):
            dE1p = q / ctotal * (0.5 * q + (n * q - q0) - (c2 + cg) * Vd[iv] + cg * Vg)
            dE1n = q / ctotal * (0.5 * q - (n * q - q0) + (c2 + cg) * Vd[iv] - cg * Vg)
            dE2p = q / ctotal * (0.5 * q - (n * q - q0) - c1 * Vd[iv] - cg * Vg)
            dE2n = q / ctotal * (0.5 * q + (n * q - q0) + c1 * Vd[iv] + cg * Vg)
            T1p.append(1 / (r1 * q**2) * (-dE1p) / (1 - np.exp(dE1p / (kb * temp))) if dE1p < 0 else 1e-1)
            T1n.append(1 / (r1 * q**2) * (-dE1n) / (1 - np.exp(dE1n / (kb * temp))) if dE1n < 0 else 1e-1)
            T2p.append(1 / (r2 * q**2) * (-dE2p) / (1 - np.exp(dE2p / (kb * temp))) if dE2p < 0 else 1e-1)
            T2n.append(1 / (r2 * q**2) * (-dE2n) / (1 - np.exp(dE2n / (kb * temp))) if dE2n < 0 else 1e-1)

        p = np.zeros(Nmax - Nmin)
        p[0] = 0.001
        for ne in range(1, Nmax - Nmin - 1):
            p[ne] = p[ne - 1] * (T2n[ne - 1] + T1p[ne - 1]) / (T2p[ne] + T1n[ne])
            p[ne] = min(max(p[ne], 1e-250), 1e250)

        p /= sum(p)
        I[iv] = q * sum(p[ne] * (T2p[ne] - T2n[ne]) for ne in range(Nmax - Nmin - 1))

    plt.figure()
    plt.plot(Vd, I)
    plt.xlabel(r"Drain voltage $V_d$ (V)")
    plt.ylabel(r"Drain current $I_d$ (A)")
    plt.title(f"Drain current, $V_g$ = {Vg:.2f} V, Temp = {temp:.2f} K")
    plt.grid()
    save_plot("I_vs_Vd.png")

# ---------- Functions for App 2 ----------
def generate_electrons_vs_vg_plot(Nmin, Nmax):
    q, cg = 1.602e-19, 1.0e-18
    Vg = [q / cg * (n + 0.5) for n in range(Nmin, Nmax + 1)]
    plt.figure("Electrons in Dot vs. Vg")
    plt.stairs(np.arange(Nmin, Nmax), Vg, fill=False)
    plt.xlabel(r"Gate voltage $V_g$ (V)")
    plt.xlim(-3, 3)
    plt.ylabel("Number of electrons stored into the dot")
    plt.title("Electrons in Dot vs. $V_g$")
    plt.grid()
    save_plot("electrons_vs_vg.png")

# ---------- Routes ----------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", plot1_generated=False, plot2_generated=False, error=None, time=int(time.time()))

@app.route("/app1", methods=["POST"])
def app1():
    try:
        # Get user inputs
        Vg = float(request.form.get("Vg"))
        temp = float(request.form.get("temp"))

        # Generate App 1 plot
        generate_plots(Vg, temp)
        app.logger.debug("App 1 plot generated successfully")

        # Render template with plot1_generated set to True
        return render_template("index.html", plot1_generated=True, plot2_generated=False, error=None, time=int(time.time()))
    except Exception as e:
        app.logger.error(f"Error in App 1: {e}")
        return render_template("index.html", plot1_generated=False, plot2_generated=False, error=f"Error generating App 1 plot: {e}", time=int(time.time()))

@app.route("/app2", methods=["POST"])
def app2():
    try:
        # Get user inputs
        Nmin = int(request.form.get("Nmin"))
        Nmax = int(request.form.get("Nmax"))

        # Generate App 2 plot
        generate_electrons_vs_vg_plot(Nmin, Nmax)
        app.logger.debug("App 2 plot generated successfully")

        # Render template with plot2_generated set to True
        return render_template("index.html", plot1_generated=False, plot2_generated=True, error=None, time=int(time.time()))
    except Exception as e:
        app.logger.error(f"Error in App 2: {e}")
        return render_template("index.html", plot1_generated=False, plot2_generated=False, error=f"Error generating App 2 plot: {e}", time=int(time.time()))

# ---------- Run the App ----------
if __name__ == "__main__":
    app.run(debug=True)
