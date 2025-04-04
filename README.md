# cluster_stats
HPC cluster statistics. Recolecta y analiza estadísticas útiles de un cluster HPC Slurm.

# Obtención de datos crudos usando el comando sacct (Slurm).
Recolección de datos crudos: Se realiza utilizando la herramienta sacct de Slurm. Slurm guarda una base de datos de todos los trabajos realizados y es posible recopilarlos utilizando este comando. Para ello se utiliza un script como el siguiente.

```
#Definición de datos a extraer
SACCT_FORMAT="State,JobIDRaw,User,Partition,AllocCPUS%-10,ReqMem,MaxRSS,ReqTRES,Submit,Start,Elapsed,End,Timelimit"
STATES="CD,TO"
sacct --format=$SACCT_FORMAT --units=M -v -P -a -S 01/24 > sacct_raw_data.dat
rm -rf tmp
grep -i state sacct_raw_data.dat > tmp
awk '/COMPLETED|TIMEOUT/{print $0}' sacct_raw_data.dat >> tmp; mv tmp sacct_raw_data.dat
```

El resultado es un archivo sacct_raw_data.dat con la siguiente forma.
```
TIMEOUT|67155|mcattaneo|cpu-x|20|60000Mn||billing=20,cpu=20,mem=60000M,node=1
COMPLETED|67156|mcattaneo|cpu-x|20|60000Mn||billing=20,cpu=20,mem=60000M,node=1
COMPLETED|67156.batch|||20|60000Mn|59611.48M|
COMPLETED|67157|mcattaneo|cpu-x|20|60000Mn||billing=20,cpu=20,mem=60000M,node=1
COMPLETED|67157.batch|||20|60000Mn|61087.04M|
TIMEOUT|67242|nguzman|mem|16|0n||billing=16,cpu=16,node=1
TIMEOUT|67243|nguzman|mem|16|0n||billing=16,cpu=16,node=1
COMPLETED|67251|escaglia|rtx4070|1|0n||billing=1,cpu=1,gres/gpu=1,node=1
COMPLETED|67251.batch|||1|0n|1342.84M|
TIMEOUT|67254|escaglia|rtx2080|2|0n||billing=1,cpu=1,gres/gpu=1,node=1
[...]
```

# Análisis y visualización.
El script parsea el archivo anterior y obtiene estadisticas por usuario y por partición de Memoria pedida, CPU pedida y si usa GPU o no, numero total de trabajos de cada usuario, uso promedio de GPU, tiempo de espera en cola, etc. Luego clasifica las categorias CPU, memoria y % de trabajos que usan GPU en bajo, medio y alto según límites arbitrarios (sujetos a revisión) y determina grupos de uso de recursos basados en los parámetros de pedidos de memoria, CPU y uso promedio de GPU. Finalmente crea una visualización interactiva en html en un archivo que puede cargarse en el navegador. 

