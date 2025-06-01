# Myocardial tagging on a user-defined phantom

This experiment is computationally intensive, involving **7 million spins** and **100 cine frames**. Therefore, we recommend running it on a sufficiently powerful machine with access to multiple GPUs, if available.

To ensure the script runs uninterrupted—especially on remote servers or during long computations—you can execute it in the background using `nohup`:

```bash
nohup julia main.jl 2>&1 &
```