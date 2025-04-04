import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.cluster.hierarchy import dendrogram
import plotly.express as px
import plotly.graph_objects as go

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
    data = []
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
        

        # Características derivadas
        # 1. Relación CPU/memoria
        cpu_mem_ratio = avg_cpus / avg_memory if avg_memory > 0 else 0

        # 2. Puntaje de uso intensivo de recursos
        resource_intensity = (avg_cpus / 40) + (avg_memory / 64000) + gpu_ratio

        # 3. Variedad de uso (qué tan balanceado es el uso de CPU vs GPU)
        # Un valor cerca de 0.5 indica uso balanceado de ambos recursos
        resource_balance = abs(0.5 - gpu_ratio)

        # 4. Escala de operación (basada en total de trabajos)
        operation_scale = np.log1p(total_jobs)  # Logaritmo para suavizar valores extremos

        # 5. Categorías de tipos de usuarios
        # Tipo 1: Uso intensivo de CPU
        cpu_intensive = avg_cpus > 8 and gpu_ratio < 0.3

        # Tipo 2: Uso intensivo de GPU
        gpu_intensive = gpu_ratio > 0.7

        # Tipo 3: Usuario de baja intensidad 
        low_intensity = avg_cpus < 4 and avg_memory < 10000 and gpu_ratio < 0.3

        # Tipo 4: Usuario balanceado de recursos
        balanced_user = not (cpu_intensive or gpu_intensive or low_intensity)

        row = {
            'user': user,
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
            'cpu_mem_ratio': cpu_mem_ratio,
            'resource_intensity': resource_intensity,
            'resource_balance': resource_balance,
            'operation_scale': operation_scale,
            'cpu_intensive': int(cpu_intensive),
            'gpu_intensive': int(gpu_intensive),
            'low_intensity': int(low_intensity),
            'balanced_user': int(balanced_user)
        }
        data.append(row)
    user_stats_df = pd.DataFrame(data)
    
    return user_stats_df

def compute_partition_statistics(df):
    """Calcula estadísticas de uso para cada partición."""
    data = [ ]
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
        
        row = {
            'partition': partition,
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
        data.append(row)
    partition_stats_df = pd.DataFrame(data)
    
    print(partition_stats_df)
    return partition_stats_df


def classify_users2(user_stats_df):
    print(user_stats_df)
    # Resumen de categorizacion de usuario segun intensidad cpu_intensive, gpu_intensive, etc.
    for cat in ['cpu_intensive', 'gpu_intensive', 'low_intensity','balanced_user']:
        cat_df = user_df[user_df[cat]==1]
        print(f"  Tipo {cat}: {', '.join(cat_df['user'].tolist())}")




def classify_users(user_stats_df):
    """Clasifica usuarios según sus patrones de uso de recursos."""
    
    # Clasificación simple basada en umbrales
    user_groups = defaultdict(list)
    
    for user in user_stats_df['user']:
        user_df = user_stats_df[user]
        # Clasificar CPU: Bajo (≤2), Medio (3-8), Alto (>8)
        cpu_class = "Low"
        if user_df['max_cpus'] > 7:
            cpu_class = "High"
        elif user_df['max_cpus'] > 2:
            cpu_class = "Medium"
        
        # Clasificar Memoria: Bajo (<4GB), Medio (4-32GB), Alto (>32GB)
        mem_class = "Low"
        if user_df['max_memory'] > 32 * 1024:
            mem_class = "High"
        elif user_df['max_memory'] > 4 * 1024:
            mem_class = "Medium"
        
        # Clasificar GPU: Sí o No
        gpu_class = "Yes" if user_df['gpu_ratio'] > 0.4 else "No"
        
        # Crear clave de grupo y añadir usuario
        group_key = f"{cpu_class}_{mem_class}_{gpu_class}"
        user_groups[group_key].append({
            'user': user_info['user'],
            'max_cpus': user_info['max_cpus'],
            'max_memory': user_info['max_memory'],
            'gpu_ratio': user_info['gpu_ratio']
        })
    
    return user_groups

def summarize_groups2(user_stats_df):
    """Crea un resumen de los grupos de usuarios."""
    for group_key in ['cpu_intensive', 'gpu_intensive', 'low_intensity','balanced_user']:
        cat_df = user_stats_df[user_stats_df[group_key]==1]
        n_users = len( cat_df['user'].tolist() )
        print(f"  Tipo {group_key}: {', '.join(cat_df['user'].tolist())}")
        print(f"  Numero de users {group_key}: {n_users}")
        print(f" ")
        
    

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

def create_3d_plot_interactive(user_stats_df):
    """Crea una gráfica 3D interactiva usando Plotly."""
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # Crear gráfica interactiva de dispersión 3D
        fig = px.scatter_3d(
            user_stats_df, 
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
    user_stats_df = compute_user_statistics(df)
    print(f"Analizados datos para {len(user_stats_df)} usuarios")
    
    # Clasificar usuarios en grupos
    # print("Clasificando usuarios según uso de recursos...")
    # user_groups = classify_users(user_stats)
    
    # Crear resumen
    print("Creando resumen de grupos de usuarios...")
    summarize_groups2(user_stats_df)
    
    # Calcular estadísticas por partición
    print("Calculando estadísticas de particion...")
    partition_stats_df = compute_partition_statistics(df)
    #print("Analizados datos para { partition_stats_df.shape[0] } particiones.")

    # Crear resumen por partición
    #print("Creando resumen por partición")
    #partition_summary = summary_partition(partition_stats)
    #print("Resumen de estadísticas de particiones.")
    #print(partition_summary)

    # Visualizar los resultados
    print("Generando visualización...")
    # Intentar crear gráfica interactiva si plotly está disponible
    try:
       create_3d_plot_interactive(user_stats_df)
    except:
        print("No se pudo crear la gráfica interactiva. Asegúrese de que plotly esté instalado.")
    
    return {
        'data_frame': df,
        'user_stats': user_stats_df,
    }


# CLUSTERING SUBROUTINES
#--------------------------------------------

def prepare_data_for_clustering(user_stats):
    """Prepara los datos para el agrupamiento."""
    # Convertir diccionario user_stats a DataFrame
    user_df = user_stats
    
    # Extraer características para clustering
    X = user_df[['avg_cpus', 'avg_memory', 'gpu_ratio']].values
    
    # Escalar los datos para normalizar las diferentes escalas
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return user_df, X, X_scaled, scaler

def kmeans_clustering(X_scaled, n_clusters=5):
    """Aplicar KMeans para agrupar usuarios."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # Calcular la inercia para diferentes números de clusters
    inertia = []
    for k in range(1, 11):
        kmeans_temp = KMeans(n_clusters=k, random_state=42)
        kmeans_temp.fit(X_scaled)
        inertia.append(kmeans_temp.inertia_)
    
    return clusters, inertia, kmeans.cluster_centers_

def dbscan_clustering(X_scaled, eps=0.5, min_samples=5):
    """Aplicar DBSCAN para agrupar usuarios."""
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(X_scaled)
    
    return clusters

def hierarchical_clustering(X_scaled, n_clusters=5, linkage='ward'):
    """Aplicar Clustering Jerárquico para agrupar usuarios."""
    hier_cluster = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    clusters = hier_cluster.fit_predict(X_scaled)
    
    # Para visualizar dendrograma (solo para propósitos analíticos)
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None, linkage=linkage)
    model = model.fit(X_scaled)
    
    return clusters, model

def gmm_clustering(X_scaled, n_components=5):
    """Aplicar Gaussian Mixture Model para agrupar usuarios."""
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_scaled)
    clusters = gmm.predict(X_scaled)
    
    # Calcular BIC para diferentes números de componentes
    bic = []
    for k in range(1, 11):
        gmm_temp = GaussianMixture(n_components=k, random_state=42)
        gmm_temp.fit(X_scaled)
        bic.append(gmm_temp.bic(X_scaled))
    
    return clusters, bic

def plot_dendrogram(model, **kwargs):
    """Crea un dendrograma para visualizar el clustering jerárquico."""
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the dendrogram
    plt.figure(figsize=(12, 8))
    dendrogram(linkage_matrix, **kwargs)
    plt.title('Dendrograma de Clustering Jerárquico')
    plt.xlabel('Usuarios')
    plt.ylabel('Distancia')
    plt.savefig('dendrogram.png')
    plt.close()

def plot_elbow_method(inertia):
    """Visualiza el método del codo para determinar el número óptimo de clusters en KMeans."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Método del Codo para Determinar el Número Óptimo de Clusters')
    plt.xlabel('Número de Clusters')
    plt.ylabel('Inercia')
    plt.savefig('elbow_method.png')
    plt.close()

def plot_bic(bic):
    """Visualiza el criterio BIC para determinar el número óptimo de componentes en GMM."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 11), bic, marker='o')
    plt.title('Criterio BIC para Determinar el Número Óptimo de Componentes')
    plt.xlabel('Número de Componentes')
    plt.ylabel('BIC')
    plt.savefig('bic_criterion.png')
    plt.close()

def create_3d_cluster_visualization(user_df, clusters, algorithm_name):
    """Crea una visualización 3D de los clusters."""
    user_df['cluster'] = clusters
    
    fig = px.scatter_3d(
        user_df, 
        x='avg_cpus', 
        y='avg_memory', 
        z='gpu_ratio',
        color='cluster',
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
        title=f'Visualización 3D de Clusters ({algorithm_name})'
    )
    
    fig.update_layout(
        scene=dict(
            xaxis_title='Utilización Promedio de CPU',
            yaxis_title='Utilización Promedio de Memoria (MB)',
            zaxis_title='Proporción de Utilización de GPU'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    fig.write_html(f'hpc_clusters_{algorithm_name}.html')
    print(f"Visualización 3D de clusters ({algorithm_name}) guardada como 'hpc_clusters_{algorithm_name}.html'")

def analyze_clusters(user_df, clusters, scaler=None, centers=None, algorithm_name=''):
    """Analiza las características de cada cluster."""
    user_df['cluster'] = clusters
    
    cluster_stats = []
    for cluster_id in sorted(user_df['cluster'].unique()):
        cluster_df = user_df[user_df['cluster'] == cluster_id]
        
        stats = {
            'cluster_id': cluster_id,
            'user_count': len(cluster_df),
            'users': list(cluster_df['user']),
            'avg_cpus_mean': cluster_df['avg_cpus'].mean(),
            'avg_memory_mean': cluster_df['avg_memory'].mean(),
            'gpu_ratio_mean': cluster_df['gpu_ratio'].mean(),
            'total_jobs_mean': cluster_df['total_jobs'].mean()
        }
        
        cluster_stats.append(stats)
    
    # Si tenemos centros de cluster (KMeans), transformar de vuelta a la escala original
    if centers is not None and scaler is not None:
        centers_original = scaler.inverse_transform(centers)
        for i, center in enumerate(centers_original):
            for j, stat, name in zip(range(3), 
                                   ['avg_cpus_center', 'avg_memory_center', 'gpu_ratio_center'],
                                   ['avg_cpus', 'avg_memory', 'gpu_ratio']):
                # Encontrar el cluster actual que corresponde al centro
                for stats in cluster_stats:
                    if stats['cluster_id'] == i:
                        stats[stat] = center[j]
    
    # Ordenar por número de usuarios (descendente)
    cluster_stats.sort(key=lambda x: x['user_count'], reverse=True)
    
    print(f"\nResumen de Clusters ({algorithm_name}):")
    for stats in cluster_stats:
        print(f"\nCluster {stats['cluster_id']} ({stats['user_count']} usuarios)")
        print(f"  CPU promedio: {stats['avg_cpus_mean']:.2f}")
        print(f"  Memoria promedio: {stats['avg_memory_mean']:.2f}")
        print(f"  Ratio GPU: {stats['gpu_ratio_mean']:.2f}")
        print(f"  Usuarios: {', '.join(stats['users'][:5])}{'...' if len(stats['users']) > 5 else ''}")
    
    return cluster_stats

def fuzzy_clustering(user_stats):
    """Aplicar clustering difuso (Fuzzy C-means) para asignar probabilidades de pertenencia a clusters."""
    # Es necesario instalar skfuzzy: pip install scikit-fuzzy
    try:
        import skfuzzy as fuzz
        
        # Preparar datos
        user_df, X, X_scaled, scaler = prepare_data_for_clustering(user_stats)
        
        # Aplicar Fuzzy C-means
        n_clusters = 5
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
            X_scaled.T, n_clusters, 2, error=0.005, maxiter=1000, init=None
        )
        
        # Transformar resultados
        membership = pd.DataFrame(
            u.T, 
            index=user_df['user'],
            columns=[f'cluster_{i}' for i in range(n_clusters)]
        )
        
        # Asignar cluster principal (el de mayor probabilidad)
        user_df['cluster'] = np.argmax(u, axis=0)
        
        # Visualizar
        create_3d_cluster_visualization(user_df, user_df['cluster'], 'FuzzyCMeans')
        
        # Analizar clusters
        analyze_clusters(user_df, user_df['cluster'], algorithm_name='Fuzzy C-means')
        
        print("\nEjemplo de probabilidades de pertenencia para algunos usuarios:")
        print(membership.head())
        
        return user_df['cluster'], membership
    
    except ImportError:
        print("skfuzzy no está instalado. Omitiendo clustering difuso.")
        return None, None

def adaptive_clustering(user_stats):
    """Encuentra automáticamente el mejor enfoque de clustering para los datos."""
    user_df, X, X_scaled, scaler = prepare_data_for_clustering(user_stats)
    
    # Probamos varios algoritmos y parámetros
    results = {}
    
    # 1. KMeans con diferentes números de clusters
    best_silhouette = -1
    best_k = 0
    from sklearn.metrics import silhouette_score
    
    for k in range(2, 8):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        silhouette = silhouette_score(X_scaled, labels)
        if silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k
    
    print(f"Mejor número de clusters KMeans según silhouette: {best_k}")
    
    # 2. DBSCAN con diferentes valores de eps
    best_dbscan_ratio = 0
    best_eps = 0
    best_min_samples = 0
    
    for eps in [0.3, 0.5, 0.7, 1.0]:
        for min_samples in [3, 5, 7]:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(X_scaled)
            # Calcular la proporción de puntos no clasificados como ruido (-1)
            n_noise = list(labels).count(-1)
            ratio_valid = 1 - (n_noise / len(labels))
            
            # Quiero al menos 2 clusters y no más del 20% como ruido
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters >= 2 and ratio_valid > best_dbscan_ratio:
                best_dbscan_ratio = ratio_valid
                best_eps = eps
                best_min_samples = min_samples
    
    print(f"Mejores parámetros DBSCAN: eps={best_eps}, min_samples={best_min_samples}")
    
    # 3. Ejecutar todos los algoritmos con los mejores parámetros
    # KMeans
    kmeans = KMeans(n_clusters=best_k, random_state=42)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    
    # DBSCAN
    if best_eps > 0:
        dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
        dbscan_labels = dbscan.fit_predict(X_scaled)
    else:
        # Usar valores predeterminados
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
    
    # Hierarchical
    hier = AgglomerativeClustering(n_clusters=best_k)
    hier_labels = hier.fit_predict(X_scaled)
    
    # GMM
    gmm = GaussianMixture(n_components=best_k, random_state=42)
    gmm_labels = gmm.fit_predict(X_scaled)
    
    # 4. Evaluar y elegir el mejor modelo
    silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
    silhouette_hier = silhouette_score(X_scaled, hier_labels)
    silhouette_gmm = silhouette_score(X_scaled, gmm_labels)
    
    # Para DBSCAN, solo calcular silhouette si hay más de un cluster
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    if n_clusters_dbscan > 1:
        # Filtrar puntos de ruido para silhouette
        mask = dbscan_labels != -1
        silhouette_dbscan = silhouette_score(X_scaled[mask], dbscan_labels[mask])
    else:
        silhouette_dbscan = -1
    
    scores = {
        'KMeans': silhouette_kmeans,
        'DBSCAN': silhouette_dbscan,
        'Hierarchical': silhouette_hier,
        'GMM': silhouette_gmm
    }
    
    print("\nPuntuaciones de silhouette para cada algoritmo:")
    for algo, score in scores.items():
        print(f"{algo}: {score:.3f}")
    
    best_algo = max(scores, key=scores.get)
    print(f"Mejor algoritmo según silhouette: {best_algo}")
    
    # Devolver los resultados del mejor algoritmo
    if best_algo == 'KMeans':
        return kmeans_labels, best_k, best_algo
    elif best_algo == 'DBSCAN':
        return dbscan_labels, {'eps': best_eps, 'min_samples': best_min_samples}, best_algo
    elif best_algo == 'Hierarchical':
        return hier_labels, best_k, best_algo
    else:  # GMM
        return gmm_labels, best_k, best_algo


def apply_clustering_strategies(user_stats):
    """Aplica varias estrategias de clustering a los datos de usuarios HPC."""
    # Preparar datos
    user_df, X, X_scaled, scaler = prepare_data_for_clustering(user_stats)
    
    # Clustering adaptativo
    print("\n=== Clustering Adaptativo ===")
    adaptive_clusters, best_params, best_algo = adaptive_clustering(user_stats)
    create_3d_cluster_visualization(user_df, adaptive_clusters, f'Adaptive_{best_algo}')
    adaptive_stats = analyze_clusters(user_df, adaptive_clusters, scaler=None, algorithm_name=f'Adaptive ({best_algo})')
    
    return {
        'adaptive': {'clusters': adaptive_clusters, 'best_params': best_params, 'best_algo': best_algo}
    }

# Función principal de clustering
def analyze_user_clustering(user_stats):
    """Analiza diferentes estrategias de clustering para usuarios HPC."""
    print("Aplicando diferentes estrategias de clustering a datos de usuarios HPC...")
    cluster_results = apply_clustering_strategies(user_stats)
    print("\nAnálisis de clustering completado.")
    
    return cluster_results



if __name__ == "__main__":
    # Ruta del archivo a los datos de contabilidad SLURM
    file_path = "../sacct_raw_data.dat"

    # Ejecutar el análisis
    results = analyze_hpc_data(file_path)

    # Extraer datos
    user_stats_df = results['user_stats']

    # Apply the clustering strategies
    cluster_results = analyze_user_clustering(user_stats_df)

