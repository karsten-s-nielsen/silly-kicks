"""TF-51 -- per-event defensive credit/debit family.

See NOTICE for full bibliographic citations (Sumpter, Soccermatics Pro module 16.3;
Bischofberger/Bauer/Baca, arXiv:2606.19931).
"""

from ._bravery import compute_bravery
from ._orchestration import compute_defensive_credits
from ._params import DEFENSIVE_CREDIT_RULES, DefensiveCreditParams

__all__ = [
    "DEFENSIVE_CREDIT_RULES",
    "DefensiveCreditParams",
    "compute_bravery",
    "compute_defensive_credits",
]
