# TrustFed

**Byzantine-robust federated learning that scores client reputation and down-weights malicious clients in real time.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Notebook](https://img.shields.io/badge/Colab-F4A261?style=flat-square&logo=googlecolab&logoColor=white)](https://colab.research.google.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

> This is the MIT-licensed mirror of [`vivek797029/TrustFed`](https://github.com/vivek797029/TrustFed),
> maintained under `Harshit-3114`. The upstream repository is unchanged.

---

## What it does

Federated learning aggregates client updates without ever seeing the clients'
data — which is exactly what makes it hard to tell a helpful client from a
malicious one. TrustFed scores every client on each round from three
independent signals, combines them into a trust value, tracks a reputation
history across rounds, and aggregates only the clients that have earned it.

```
client update ──┬──► CAS  cosine alignment against the server's root set   (weight 0.4)
                ├──► LCS  loss change on the root set                      (weight 0.3)
                └──► NCS  norm consistency with the global model             (weight 0.3)
                                    │
                                    ▼
                        trust = softmax(score / T)ᵢ, floored at 0.02
                                    │
                                    ▼
                     reputation = γ · reputation + (1 − γ) · trust
                                    │
                                    ▼
                  aggregate the top 70 % of clients by reputation
```

## The three signals

| Signal | Full name | What it measures | Formula |
|---|---|---|---|
| **CAS** | Cosine Alignment Score | Does the client's update point the same way as the server's own root-set step? | `(cos + 1) / 2`, mapped to `[0, 1]` |
| **LCS** | Loss Change Score | Did the client's update actually improve the server's loss on held-out root data? | `clamp((loss_before − loss_after) / loss_before, 0, 1)` |
| **NCS** | Norm Consistency Score | Is the update's magnitude in line with its direction, or has it been inflated? | `exp(−max(0, ratio − 1))` |

A client that is benign but simply different still gets a fair score: CAS
removes the sign ambiguity, and NCS penalises scaled-up updates rather than
merely large ones.

## What is compared against

| Method | Key | |
|---|---|---|
| FedAvg | `fedavg` | No defence — the reference point |
| Multi-Krum | `krum` | Selects by geometric median distance |
| Trimmed Mean | `trimmed_mean` | Drops the extreme 10 % of updates |
| FLTrust | `fltrust` | Server root set, no reputation history |
| FoolsGold | `foolsgold` | Targets Sybil-style clients |
| **TrustFed** | `medtrace` | This work |

## The attacks

`none`, `gaussian`, `sign_flip`, `label_flip`, `backdoor`

The backdoor applies a 4-pixel trigger patch to the bottom-right corner and
targets class 0. It is measured with **Attack Success Rate** — the fraction
of triggered samples the aggregated model predicts as the target class.

## Experimental setup

| | |
|---|---|
| Clients | 20, of which 2 are malicious (10 %) |
| Rounds | 50 |
| Local epochs | 3, learning rate 0.01, batch size 64 |
| Heterogeneity | Dirichlet α = 0.5, swept over {0.1, 0.5, 1.0} |
| Seeds | 42, 123, 777, 2024, 9999 — five, so results carry a standard deviation |
| Server root set | 200 samples, used for CAS and LCS |
| Aggregation | Top 70 % of clients by reputation |
| Malicious threshold | Trust < 0.35 |
| Clipping | Gradient norm 1.0, update norm 5.0 |

**Datasets:** CIFAR-10, Fashion-MNIST, FEMNIST (per-writer, as a natural
non-IID split), and Shakespeare for a text task. Models: `CIFARCNN`,
`FMNISTNet`, `ShakespeareLSTM`, all using GroupNorm so no batch statistics
leak between clients.

## Metrics

Accuracy, loss, backdoor ASR, and **detection precision / recall / F1** for
identifying malicious clients — plus the trust and reputation traces
themselves.

## Ablations

- **Component** — remove CAS, LCS, NCS or reputation in turn
- **Malicious ratio** — 10 %, 20 %, 30 %
- **Dirichlet α** — 0.1, 0.5, 1.0
- **Hyperparameters** — γ, the top-k fraction, and the trust threshold

The full run produces 9 tables and 18 figures.

## Files

| File | What it is |
|---|---|
| `TrustFed_FL_v6.ipynb` | The 67-cell Colab notebook — run it top to bottom |
| `TrustFed_FL_v6.py` | The same system as a script, for running outside Colab |
| `MedTrace_FL_v6.ipynb` | The medical-imaging federated learning variant |
| `MedTrace_FL_v6.py` | The same, as a script |

## Running it

The notebooks are designed for Google Colab on a GPU (T4 or better).

1. Open `TrustFed_FL_v6.ipynb` in Colab.
2. **Run Step 0 first.** It mounts Google Drive so checkpoints and CSV results
   survive a disconnect.
3. Run the remaining cells top to bottom.

Outside Colab:

```bash
pip install torch torchvision matplotlib numpy
python TrustFed_FL_v6.py
```

Checkpoints, per-round CSV logs and the generated figures are written to
Google Drive when it is mounted, and to the working directory otherwise.

## Reproducibility

Every experiment is seeded. `SEEDS = [42, 123, 777, 2024, 9999]` and the
aggregation functions are deterministic given a seed, so a reported mean ±
standard deviation can be reproduced exactly.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Harshit Shyamsukha.
