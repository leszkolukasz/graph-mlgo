import os
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

def plot_benchmark_results(checkpoint_dirs: list[str]):
    """
    Wczytuje zyski agenta i LLVM z wielu checkpointów i generuje interaktywny Violin Plot.
    """
    fig = go.Figure()
    
    # 1. Najpierw dodajemy baseline LLVM
    # Zakładamy, że LLVM zachowuje się tak samo we wszystkich testach, 
    # więc bierzemy plik llvm_gains.npy z pierwszego podanego katalogu
    if not checkpoint_dirs:
        print("Brak katalogów do przetworzenia.")
        return
        
    first_dir = checkpoint_dirs[0]
    llvm_path = os.path.join(first_dir, "llvm_gains.npy")
    
    if os.path.exists(llvm_path):
        llvm_gains = np.load(llvm_path)
        
        # Obliczamy metryki dla lepszego opisu na wykresie
        llvm_mean = np.mean(llvm_gains)
        llvm_median = np.median(llvm_gains)
        
        fig.add_trace(go.Violin(
            y=llvm_gains,
            name="<b>Baseline: LLVM (Oz)</b>", # Pogrubione dla wyróźnienia
            box_visible=True, # Pokazuje wewnątrz klasyczny box-plot
            meanline_visible=True, # Rysuje linię średniej
            fillcolor='rgba(128, 128, 128, 0.5)', # Szary dla baseline'u
            line_color='gray',
            hovertext=f"Średnia: {llvm_mean:.1f}<br>Mediana: {llvm_median:.1f}",
            points="outliers", # Pokazuje tylko punkty odstające, by nie zamulić przeglądarki
            jitter=0.05
        ))
    else:
        print(f"OSTRZEŻENIE: Nie znaleziono pliku {llvm_path}")

    # 2. Pętla po wszystkich podanych modelach (Agentach)
    colors = ['rgba(31, 119, 180, 0.6)', 'rgba(255, 127, 14, 0.6)', 
              'rgba(44, 160, 44, 0.6)', 'rgba(214, 39, 40, 0.6)', 
              'rgba(148, 103, 189, 0.6)']

    for i, ckpt_dir in enumerate(checkpoint_dirs):
        agent_path = os.path.join(ckpt_dir, "agent_gains.npy")
        
        if not os.path.exists(agent_path):
            print(f"Pomijanie {ckpt_dir} - brak pliku agent_gains.npy")
            continue
            
        agent_gains = np.load(agent_path)
        
        # Pobieramy nazwę folderu końcowego jako nazwę modelu do legendy
        model_name = Path(ckpt_dir).name
        
        agent_mean = np.mean(agent_gains)
        agent_median = np.median(agent_gains)
        
        # Wybieramy kolor z palety (lub zapętlamy)
        color = colors[i % len(colors)]
        
        fig.add_trace(go.Violin(
            y=agent_gains,
            name=f"Agent: {model_name}",
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            line_color=color.replace('0.6', '1.0'), # Mniej przezroczyste obramowanie
            hovertext=f"Średnia: {agent_mean:.1f}<br>Mediana: {agent_median:.1f}",
            points="outliers",
            jitter=0.05
        ))

    # 3. Kształtowanie layoutu
    fig.update_layout(
        title="<b>Porównanie skuteczności inlinowania: RL Agent vs LLVM Oz</b><br><sup>Rozkład zaoszczędzonych bajtów względem kodu bez inlinowania (Więcej = Lepiej)</sup>",
        yaxis_title="Zaoszczędzone bajty (Gain)",
        xaxis_title="Ewaluowany Model",
        violingap=0.2,
        violingroupgap=0.1,
        violinmode='group',
        template='plotly_white',
        font=dict(size=14),
        hovermode="closest",
        # Rysuje poziomą linię na wysokości Y=0 dla ułatwienia odczytu (granica "puchnięcia" kodu)
        shapes=[dict(
            type='line',
            y0=0, y1=0,
            x0=-0.5, x1=len(checkpoint_dirs) + 0.5,
            line=dict(color='Red', width=2, dash='dash')
        )]
    )

    fig.show()


if __name__ == "__main__":
    checkpoints_to_compare = [
        "./models/ppo/final_ppo_sparse",
        "./models/ppo/final_ppo_256",
    ]
    
    plot_benchmark_results(checkpoints_to_compare)