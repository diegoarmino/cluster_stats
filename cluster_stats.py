import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

def parse_slurm_data(file_path):
    """Analiza datos de contabilidad SLURM desde un archivo de texto."""
    # Leer el archivo
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    # Inicializar listas para almacenar datos analizados
    parsed_records = []
    
    # Analizar cada línea
    for line in lines:
        parts = line.split('|')
        
        # Omitir líneas con partes insuficientes o trabajos por lotes
        if len(parts) < 5 or not 'COMPLETED' in parts[0] or '.batch' in parts[1]:
            continue
            
        job_id = parts[1]
        user = parts[2]
        partition = parts[3]
        
        # Extraer asignación de CPU
        alloc_cpus = 0
        if parts[4] and parts[4].isdigit():
            alloc_cpus = int(parts[4])
        
        # Extraer asignación de memoria
        alloc_mem = 0
        if parts[5]:
            if 'Mn' in parts[5]:
                alloc_mem = int(parts[5].replace('Mn', ''))
            elif '0n' in parts[5]:
                alloc_mem = int('64000')
        
        # Determinar si el trabajo usó GPU
        has_gpu = False
        if partition and any(gpu_type in partition for gpu_type in ['gpu', 'A100', 'rtx']):
            has_gpu = True
        
        # Determinar tiempo de espera.
        if parts[8] and parts[9]:
            # Tiempo de submit del trabajo.
            submit = datetime.fromisoformat(parts[8])
            # Tiempo de initio del trabajo
            end = datetime.fromisoformat(parts[9])
            # Tiempo de espera en la cola.
            wait = end - submit

        # Agregar el registro
        parsed_records.append({
            'job_id': job_id,
            'user': user,
            'partition': partition,
            'alloc_cpus': alloc_cpus,
            'has_gpu': has_gpu,
            'alloc_mem': alloc_mem,
            'wait': wait,
        })
    
    # Convertir a DataFrame
    df = pd.DataFrame(parsed_records)
    return df

def compute_user_statistics(df):
    """Calcula estadísticas de uso para cada usuario."""
    user_stats = {}
    print(df)
    for user in df['user'].unique():
        user_df = df[df['user'] == user]
        
        # Estadísticas básicas
        total_jobs = len(user_df)
        gpu_jobs = user_df['has_gpu'].sum()
        cpu_jobs = total_jobs - gpu_jobs
        
        # Estadísticas de CPU
        total_cpus = user_df['alloc_cpus'].sum()
        max_cpus = user_df['alloc_cpus'].max()
        avg_cpus = total_cpus / total_jobs if total_jobs > 0 else 0
        
        # Estadísticas de memoria
        total_memory = user_df['alloc_mem'].sum()
        max_memory = user_df['alloc_mem'].max()
        avg_memory = total_memory / total_jobs if total_jobs > 0 else 0
        
        # Proporción de GPU
        gpu_ratio = gpu_jobs / total_jobs if total_jobs > 0 else 0
        
        user_stats[user] = {
            'total_jobs': total_jobs,
            'gpu_jobs': gpu_jobs,
            'cpu_jobs': cpu_jobs,
            'total_cpus': total_cpus,
            'max_cpus': max_cpus,
            'avg_cpus': avg_cpus,
            'total_memory': total_memory,
            'max_memory': max_memory,
            'avg_memory': avg_memory,
            'gpu_ratio': gpu_ratio
        }
    
    return user_stats

def compute_partition_statistics(df):
    """Calcula estadísticas de uso para cada partición."""
    partition_stats = {}
    for partition in df['partition'].unique():
        partition_df = df[df['partition'] == partition]
        
        # Estadísticas básicas
        total_jobs = len(partition_df)
        gpu_jobs = partition_df['has_gpu'].sum()
        cpu_jobs = total_jobs - gpu_jobs
        
        # Estadísticas de CPU
        total_cpus = partition_df['alloc_cpus'].sum()
        max_cpus = partition_df['alloc_cpus'].max()
        avg_cpus = total_cpus / total_jobs if total_jobs > 0 else 0
        
        # Estadísticas de memoria
        total_memory = partition_df['alloc_mem'].sum()
        max_memory = partition_df['alloc_mem'].max()
        avg_memory = total_memory / total_jobs if total_jobs > 0 else 0
        
        # Proporción de GPU
        gpu_ratio = gpu_jobs / total_jobs if total_jobs > 0 else 0
        
        # Tiempo de espera
        total_wait = partition_df['wait'].sum()
        max_wait = partition_df['wait'].max()
        mean_wait = partition_df['wait'].mean()
        
        partition_stats[partition] = {
            'total_jobs': total_jobs,
            'gpu_jobs': gpu_jobs,
            'cpu_jobs': cpu_jobs,
            'total_cpus': total_cpus,
            'max_cpus': max_cpus,
            'avg_cpus': avg_cpus,
            'total_memory': total_memory,
            'max_memory': max_memory,
            'avg_memory': avg_memory,
            'gpu_ratio': gpu_ratio,
            'max_wait': max_wait,
            'mean_wait': mean_wait,
        }
    
    return partition_stats

def summary_partition(partition_stats):
    # Convertir diccionario user_stats a DataFrame
    data = []
    for partition, stats in partition_stats.items():
        row = {
            'partition': partition,
            'avg_cpus': stats['avg_cpus'],
            'avg_memory': stats['avg_memory'],
            'gpu_ratio': stats['gpu_ratio'],
            'total_jobs': stats['total_jobs'],
            'max_wait': stats['max_wait'],
            'mean_wait': stats['mean_wait'],
        }
        data.append(row)

    # Crear DataFrame
    partition_df = pd.DataFrame(data)
    return partition_df


def classify_users(user_stats):
    """Clasifica usuarios según sus patrones de uso de recursos."""
    # Preparar datos para agrupamiento
    user_data = []
    for user, stats in user_stats.items():
        user_data.append({
            'user': user,
            'avg_cpus': stats['avg_cpus'],
            'max_cpus': stats['max_cpus'],
            'avg_memory': stats['avg_memory'],
            'max_memory': stats['max_memory'],
            'gpu_ratio': stats['gpu_ratio']
        })
    
    # Clasificación simple basada en umbrales
    user_groups = defaultdict(list)
    
    for user_info in user_data:
        # Clasificar CPU: Bajo (≤2), Medio (3-8), Alto (>8)
        cpu_class = "Low"
        if user_info['max_cpus'] > 7:
            cpu_class = "High"
        elif user_info['max_cpus'] > 2:
            cpu_class = "Medium"
        
        # Clasificar Memoria: Bajo (<4GB), Medio (4-32GB), Alto (>32GB)
        mem_class = "Low"
        if user_info['max_memory'] > 32 * 1024:
            mem_class = "High"
        elif user_info['max_memory'] > 4 * 1024:
            mem_class = "Medium"
        
        # Clasificar GPU: Sí o No
        gpu_class = "Yes" if user_info['gpu_ratio'] > 0.4 else "No"
        
        # Crear clave de grupo y añadir usuario
        group_key = f"{cpu_class}_{mem_class}_{gpu_class}"
        user_groups[group_key].append({
            'user': user_info['user'],
            'max_cpus': user_info['max_cpus'],
            'max_memory': user_info['max_memory'],
            'gpu_ratio': user_info['gpu_ratio']
        })
    
    return user_groups

def summarize_groups(user_groups):
    """Crea un resumen de los grupos de usuarios."""
    summary = []
    
    for group_key, users in user_groups.items():
        cpu_class, mem_class, gpu_class = group_key.split('_')
        
        summary.append({
            'group': group_key,
            'cpu_usage': cpu_class,
            'memory_usage': mem_class,
            'gpu_usage': gpu_class,
            'user_count': len(users),
            'users': [u['user'] for u in users]
        })
    
    # Ordenar por número de usuarios (descendente)
    summary.sort(key=lambda x: x['user_count'], reverse=True)
    return summary

def create_3d_plot_interactive(user_stats):
    """Crea una gráfica 3D interactiva usando Plotly."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # Convertir diccionario user_stats a DataFrame
        data = []
        for user, stats in user_stats.items():
            row = {
                'user': user,
                'avg_cpus': stats['avg_cpus'],
                'avg_memory': stats['avg_memory'],
                'gpu_ratio': stats['gpu_ratio'],
                'total_jobs': stats['total_jobs']
            }
            data.append(row)

        # Crear DataFrame
        user_df = pd.DataFrame(data)
        print(user_df)

        # Crear gráfica interactiva de dispersión 3D
        fig = px.scatter_3d(
            user_df, 
            x='avg_cpus', 
            y='avg_memory', 
            z='gpu_ratio',
            color='gpu_ratio',
            size='total_jobs',
            size_max=50,
            hover_name='user',
            hover_data={
                'user': True,
                'avg_cpus': True,
                'avg_memory': True,
                'gpu_ratio': True,
                'total_jobs': True
            },
            labels={
                'avg_cpus': 'Utilización Promedio de CPU',
                'avg_memory': 'Utilización Promedio de Memoria (MB)',
                'gpu_ratio': 'Proporción de Utilización de GPU',
                'total_jobs': 'Total de Trabajos'
            },
            title='Visualización 3D Interactiva de Utilización de Recursos de Usuarios HPC'
        )
        
        # Mejorar diseño
        fig.update_layout(
            scene=dict(
                xaxis_title='Utilización Promedio de CPU',
                yaxis_title='Utilización Promedio de Memoria (MB)',
                zaxis_title='Proporción de Utilización de GPU'
            ),
            margin=dict(l=0, r=0, b=0, t=30),
            coloraxis_colorbar=dict(
                title='Proporción GPU'
            )
        )
        
        # Guardar como HTML
        fig.write_html('hpc_3d_visualization_interactive.html')
        print("Gráfica 3D interactiva guardada como 'hpc_3d_visualization_interactive.html'")
        
        return True
    except ImportError:
        print("Plotly no está instalado. Omitiendo gráfica interactiva.")
        return False



def analyze_hpc_data(file_path):
    """Analiza datos de uso del clúster HPC e identifica grupos de usuarios."""
    # Analizar los datos
    print("Analizando datos de contabilidad SLURM...")
    df = parse_slurm_data(file_path)
    print(f"Analizados {len(df)} registros de trabajos")
    
    # Calcular estadísticas de usuario
    print("Calculando estadísticas de usuario...")
    user_stats = compute_user_statistics(df)
    print(f"Analizados datos para {len(user_stats)} usuarios")
    
    # Clasificar usuarios en grupos
    print("Clasificando usuarios según uso de recursos...")
    user_groups = classify_users(user_stats)
    
    # Crear resumen
    print("Creando resumen de grupos de usuarios...")
    summary = summarize_groups(user_groups)
    
    # Calcular estadísticas por partición
    print("Calculando estadísticas de particion...")
    partition_stats = compute_partition_statistics(df)
    print("Analizados datos para {len(partition_stats)} particiones.")

    # Crear resumen por partición
    print("Creando resumen por partición")
    partition_summary = summary_partition(partition_stats)
    print("Resumen de estadísticas de particiones.")
    print(partition_summary)

    # Visualizar los resultados
    print("Generando visualización...")
    # Intentar crear gráfica interactiva si plotly está disponible
    try:
       create_3d_plot_interactive(user_stats)
    except:
        print("No se pudo crear la gráfica interactiva. Asegúrese de que plotly esté instalado.")
    
    return {
        'data_frame': df,
        'user_stats': user_stats,
        'user_groups': user_groups,
        'summary': summary,
    }

if __name__ == "__main__":
    # Ruta del archivo a los datos de contabilidad SLURM
    file_path = "../sacct_raw_data.dat"
    
    # Ejecutar el análisis
    results = analyze_hpc_data(file_path)
    
    # Imprimir el resumen de grupos de usuarios
    print("\nResumen de Grupos de Usuarios del Clúster HPC:")
    for group in results['summary']:
        print(f"\n{group['group']} (CPU: {group['cpu_usage']}, Memoria: {group['memory_usage']}, GPU: {group['gpu_usage']})")
        print(f"  Número de usuarios: {group['user_count']}")
        print(f"  Usuarios: {', '.join(group['users'])}")
