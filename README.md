# Mean Exit Times — accompanying code base to the seminar talk

Talk on **Chapter 12** of Higham & Kloeden, *An Introduction to the Numerical
Simulation of Stochastic Differential Equations* (SIAM, 2021), for the TUM
seminar *Numerics of Stochastic Differential Equations*

Running example everywhere: geometric Brownian motion
`dX = mu X dt + sigma X dW` on the interval `(a,b) = (0.5, 2)`.

## Quick start

```bash
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```
