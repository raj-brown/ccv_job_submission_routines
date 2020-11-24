## Scripts to run the Machine Learning codes on OSCAR especially with GPU(s)
* `script0.sbatch` is a `SBATCH` script to run a `tensoflow` based code.
* `script1` is a batch script to run a `tensorflow` code with GPU of specific architetcture.  
This batch script runs of GPU of `titanrtx` make.  
To know the features of GPU please do `nodestat` on the terminal and choose any word froms feature column.  
e.g. `quadrortx`, `v100`

* `script2` is a batch script to run a `pytorch` code with GPU of specific architetcture. 
