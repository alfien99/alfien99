# ── Strategy constants ────────────────────────────────────────────────────────
# Import these instead of scattering raw numbers throughout each strategy file.
#
# Usage:
#   from utils.constants import VALUE_AREA_PCT, POC_EXIT_FRAC, VAH_EXIT_FRAC

# Volume profile
VALUE_AREA_PCT  = 0.70   # 70% of volume defines the value area (industry standard)
VP_BINS         = 100    # price bins used when building the volume profile

# Exit fractions (applied sequentially to the open position)
POC_EXIT_FRAC   = 0.50   # close 50% of position when price reaches POC
VAH_EXIT_FRAC   = 0.70   # close 70% of remainder when price reaches VAH
# The final remainder closes at the extension target

# Extension target multiplier (applied to VAH - POC range beyond VAH)
EXT_MULT        = 1.0

# Risk / position sizing
RISK_PCT        = 0.02   # risk 2% of capital per trade
MAX_POS_PCT     = 0.40   # never deploy more than 40% of capital in one trade

# Stop placement
STOP_PCT        = 0.015  # default stop 1.5% below VAL (or above VAH for shorts)

# Entry
ENTRY_BUFFER    = 0.005  # limit order placed 0.5% inside the VAL zone
FLIP_TOLERANCE  = 0.03   # VAL(t) must be within 3% of VAH(t-1) to call a flip

# Trend filter
EMA_FAST        = 50     # short EMA used in flip-long strategy
EMA_SLOW        = 200    # long EMA used in staircase strategy

# Transaction costs
COMMISSION      = 0.0005  # 0.05% per side
SLIP_MKT        = 0.0010  # 0.10% slippage on market orders
SLIP_LIM        = 0.0005  # 0.05% slippage on limit fills

# Walk-forward
INIT_CAP        = 10_000  # starting capital ($)
TRAIN_YRS       = 2       # training window length (years)
TEST_YRS        = 1       # out-of-sample test window length (years)
