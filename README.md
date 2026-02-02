# IIIT-Guwahati-Institute-GPU-Setup
This guide compiles everything you need to set up a working environment on the IIIT Guwahati's GPU server (the shared Linux cluster with multiple NVIDIA GPUs).
It is written based on real troubleshooting experience so that you (and future students) don’t have to go through the same trial-and-error.


## Take access of the GPU
1. Contact ICT and take access of the GPU
2. They will provide Username, Password, and Server IP

Username: xxxx.yyyy            
Password: XYZ    
Server IP: 1xx.xx.x.xx

3. Then install [Bitwise SSH Client](https://bitvise.com/ssh-client-download). 
4. Provide <Server IP> at Host, <Username> at username and port = 22 and then Log in. 
5. Go to the `New terminal console` and give the password. Now it will show,

```
[username@localhost]$
```
6. See the location using `pwd`, create directory `mkdir <Project>`, change directory `cd <Project>`.



## Step 1: Install Conda (User-Level Miniconda – Required)
Do not try dnf install conda or accept the “Install package ‘conda’?” prompt — it will fail due to no authentication.

Go to your home directoryBashcd ~
Download Miniconda (Linux 64-bit)Bash

```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.shIf
```
if wget is missing:

```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

Run the installer 
```
Miniconda3-latest-Linux-x86_64.sh
```

Press Enter to scroll license

Type yes to accept license

Press Enter to accept default location (~/miniconda3)
When asked 
```
Do you wish the installer to initialize Miniconda3?[yes|no]
``` 
→ type `yes`

Reload shell
```
source ~/.bashrc
```

Verify
```
conda --version
```

You should see something like conda 24.x.x.

If “command not found” still appears

```
export PATH=$HOME/miniconda3/bin:$PATH
```

```
source ~/.bashrc
```

## Step 2: Accept Anaconda Terms of Service (One-Time)
Newer Conda versions require this before creating environments.
Bash
```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
```
```
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

You should see a confirmation message like:
```
Accepted Terms of Service for channel ...
```




## Step 3: Create a Conda Environment
Never name your environment <Project> or anything that matches your project folder name — it can conflict with the base environment path. If face any problem then try to create environment using other name.

Recommended names: vlm, dl, gpuenv, myenv .....

```
conda create -n vlm python=3.10 -y
```
```
conda activate vlm
```
Your prompt should now show (vlm) at the beginning.
Alternative (path-based, no name conflict):

```
conda create -p ./vlm_env python=3.10 -y
```
```
conda activate ./vlm_env
```

Verify
```
python --version
```

Expected:
```
Python 3.10.x
```

## Step 4: Check Available GPUs and Choose One
Run:
```
nvidia-smi
```
Look at the table:
Choose a GPU with very low memory usage (ideally only 4–100 MiB, just Xorg).

Reserve your chosen GPU:
```
export CUDA_VISIBLE_DEVICES=0    # for GPU 0
```
or
```
export CUDA_VISIBLE_DEVICES=2    # for GPU 2
```
Do this every time before running Python code or starting Jupyter.

## Step 5: Install Required Packages (Inside Your Env)
Activate env first:
```
conda activate vlm
```
PyTorch with CUDA (recommended way):
```
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```
(Adjust CUDA version if nvidia-smi shows different, e.g., 11.8)
Common ML packages:
```
pip install transformers accelerate datasets jupyterlab
```
or via conda
```
conda install -c conda-forge jupyterlab -y
```

## Step 6: Start Jupyter Notebook with GPU
```
conda activate vlm
export CUDA_VISIBLE_DEVICES=0        # choose your GPU
jupyter notebook --no-browser
```
Jupyter will try ports 8888 → higher if busy.
It will print something like:
```
http://localhost:88XX/tree?token=xxxxxxxxxxxxxxxx
```
Do NOT press Ctrl+C — that stops the server.

## Step 7: Access the Jupyter Interface
### Case A: You have SSH access to Bitwise (recommended)

Note the port Jupyter printed (e.g., 8856).
Open a new terminal on your laptop (Windows: PowerShell / CMD / Windows Terminal).
Run:
```
ssh -L 8856:localhost:8856 your_username@<server_ip>
```
Replace:
your_username → your username given by the Institute 
<server_ip> → the actual server address (Provided By the Institute)

Keep this SSH terminal open.
On your laptop browser, go to:
`http://localhost:8856` Enter the token if asked → you get full Jupyter Notebook interface with GPU access. Copy the Token from the terminal and Paste it in Token section in the notebook. 

### Case B: Web portal only (no direct SSH)
Unfortunately, port forwarding won’t work. You cannot open Jupyter in your local browser.
Options:

Use terminal-only workflow (recommended for training):
```
conda activate vlm
export CUDA_VISIBLE_DEVICES=0
python your_script.py
```

Ask admin/lab in-charge:
“Can we get Jupyter Notebook access through the web portal?”
Some institutes add a “Jupyter” button later.
Alternative: Use VS Code Remote-SSH (if allowed) — gives full IDE with GPU.


### Step 8: First Cell in Your Notebook (GPU Verification)
Create a new notebook → Python (vlm) kernel.
Cell 1:
```
!pip install torch
import torch
import sys

print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    # Test operation
    x = torch.rand(5, 5).cuda()
    print("Test tensor on GPU:\n", x)
else:
    print("No GPU — check CUDA_VISIBLE_DEVICES")
```    
Run it → you should see True and your chosen GPU name.

Bonus Tips

Long-running jobs: Use tmux so they survive logout.Bashtmux new -s myjob
run your training
Ctrl+B then D to detach
tmux attach -t myjob   # to reattach
Never install packages in (base) — always activate your env first.
Environment export (for reproducibility):Bashconda env export > environment.yml
Kill your own stuck processes (if GPU is hung):Bashnvidia-smi                  # note PID
kill -9 <PID>

You now have a fully working GPU + Conda + Jupyter setup on Bitwise.
Expected outcome:

Python & PyTorch versions are displayed.
GPU availability = True.
Lists all GPU devices.
Small tensor is allocated on GPU.

### Best Practices

Always run the GPU check before starting any heavy training.
Use `export CUDA_VISIBLE_DEVICES=<GPU_ID>` to select a specific GPU.
For long-running jobs, use tmux:

Bashtmux new -s mysession
# start Jupyter or training
Ctrl+B then D to detach

Avoid pressing Ctrl+C in the terminal with running Jupyter.
Install all dependencies in the same Conda environment to avoid conflicts.

### Accessing Later

Re-login to Bitwise via SSH or web portal.
Activate Conda environment:

```
conda activate vlm
```
Start Jupyter:

```
jupyter notebook --no-browser
```

Use the same port forwarding procedure to open it on your local browser.

## References / Useful Commands

|Command | Purpose |
|---------------------|-------------|
|conda activate vlm |           Activate project environment |
|conda create -n vlm python=3.10| Create a new Conda env|
|jupyter notebook --no-browser --port=8888 | Start Jupyter Notebook|
|export CUDA_VISIBLE_DEVICES=0         |Select GPU|
|nvidia-smi               | Check GPU status|
|tmux new -s session            | Keep long-running tasks alive|
