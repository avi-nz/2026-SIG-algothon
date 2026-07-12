import numpy as np

nInst = 51

LOOKBACK = 20
QUINTILE_FRAC = 0.2
MAX_DLR_POS = 9500.0


def getMyPosition(prcSoFar):
    (nins, nt) = prcSoFar.shape

    if nt < LOOKBACK + 2:
        return np.zeros(nins, dtype=int)

    logp = np.log(prcSoFar)
    trail_ret = logp[:, -1] - logp[:, -1 - LOOKBACK]

    q = max(1, int(nins * QUINTILE_FRAC))
    order = np.argsort(trail_ret)
    losers = order[:q]     # recent underperformers -> LONG (reversal)
    winners = order[-q:]   # recent outperformers   -> SHORT (reversal)

    dollarPos = np.zeros(nins)
    dollarPos[losers] = MAX_DLR_POS
    dollarPos[winners] = -MAX_DLR_POS

    curPrices = prcSoFar[:, -1]
    shares = (dollarPos / curPrices).astype(int)

    return shares