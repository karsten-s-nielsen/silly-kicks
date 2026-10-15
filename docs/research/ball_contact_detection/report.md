# TF-79 ball-contact detection validation

Corpus: 966 matches, providers ['gradientsports', 'idsse', 'skillcorner'].
Reported, not gated (no accuracy threshold; spec sec. 8, amended).

SPADL event timestamps LAG the physical ball contact, so a tight symmetric window measures
event-feed misalignment, not detector quality. Hence the characterisation below, not one number.

## Per provider

Authoritative read: each provider is scored ONLY on its own matches, so these blocks are not
diluted by the provider mix. SkillCorner (the primary production provider) is listed first.

### skillcorner
- contacts: 3851865 (resolved 3136265, events 1048765)
- contacts/min: 43.25532428593086
- unresolved rate: 0.18578013507742352
- signed event->contact offset (s): {'p10': -0.28999999999973625, 'p25': -0.10000000000000142, 'p50': 0.0, 'p75': 0.1999999999998181, 'p90': 1.099999999999909}
- agreement vs tolerance: {'agree_within_0.2s': 0.6326774825628239, 'agree_within_0.5s': 0.7897860817246952, 'agree_within_1.0s': 0.8449223610627739}
- confidence calibration (event-match rate by bucket): {'low': {'n': 59075, 'event_match_rate': 0.047278882776132035}, 'mid': {'n': 2567619, 'event_match_rate': 0.30614744633062774}, 'high': {'n': 509571, 'event_match_rate': 0.4007449403517861}}
- velocity-change |dv| (m/s): {'p10': 0.4122459840197182, 'p25': 0.761577210454125, 'p50': 1.4142675153183657, 'p75': 2.537882547030158, 'p90': 4.162899110599176}

### gradientsports
- contacts: 69430 (resolved 54031, events 90053)
- contacts/min: 6.90518032734712
- unresolved rate: 0.22179173268039754
- signed event->contact offset (s): {'p10': -236.66582950000006, 'p25': -54.868946000000136, 'p50': 0.939443500000209, 'p75': 49.739931500000154, 'p90': 221.6415939999997}
- agreement vs tolerance: {'agree_within_0.2s': 0.017745105660000222, 'agree_within_0.5s': 0.04615060020210321, 'agree_within_1.0s': 0.10723684941090247}
- confidence calibration (event-match rate by bucket): {'low': {'n': 1642, 'event_match_rate': 0.05968331303288672}, 'mid': {'n': 38071, 'event_match_rate': 0.07914160384544666}, 'high': {'n': 14318, 'event_match_rate': 0.11056013409694092}}
- velocity-change |dv| (m/s): {'p10': 0.21191180417016547, 'p25': 0.5244668756864586, 'p50': 1.1411879663080655, 'p75': 2.500697522276286, 'p90': 4.528446064345513}

### idsse
- contacts: 25818 (resolved 22043, events 8581)
- contacts/min: 38.62599302823118
- unresolved rate: 0.1462158184212565
- signed event->contact offset (s): {'p10': -2.3694000000000526, 'p25': -0.8307499999997958, 'p50': -0.09999999999990905, 'p75': 0.4169999999999163, 'p90': 1.244699999999966}
- agreement vs tolerance: {'agree_within_0.2s': 0.22806199743619623, 'agree_within_0.5s': 0.44691760867031816, 'agree_within_1.0s': 0.6520219088684303}
- confidence calibration (event-match rate by bucket): {'low': {'n': 79, 'event_match_rate': 0.0759493670886076}, 'mid': {'n': 7866, 'event_match_rate': 0.18726163234172388}, 'high': {'n': 14098, 'event_match_rate': 0.23329550290821394}}
- velocity-change |dv| (m/s): {'p10': 0.7120154923233116, 'p25': 1.833311717267509, 'p50': 5.038765821605108, 'p75': 10.584129256701376, 'p90': 17.878144140903654}

## Pooled across all providers

_Provider-mix dependent: dominated by whichever provider contributes the most matches (here SkillCorner). This is NOT a detector-quality score -- read the per-provider blocks above._

### overall (pooled)
- contacts: 3947113 (resolved 3212339, events 1147399)
- contacts/min: 39.56106020205131
- unresolved rate: 0.18615479212274896
- signed event->contact offset (s): {'p10': -0.3999999999996362, 'p25': -0.10000000000002274, 'p50': 0.0, 'p75': 0.20000000000027285, 'p90': 1.9000000000003183}
- agreement vs tolerance: {'agree_within_0.2s': 0.5813888629848901, 'agree_within_0.5s': 0.7288580519941189, 'agree_within_1.0s': 0.7855828704748741}
- confidence calibration (event-match rate by bucket): {'low': {'n': 60796, 'event_match_rate': 0.047651161260609254}, 'mid': {'n': 2613556, 'event_match_rate': 0.30248290069162476}, 'high': {'n': 537987, 'event_match_rate': 0.38863392609858605}}
- velocity-change |dv| (m/s): {'p10': 0.4059600847808061, 'p25': 0.7615696970597584, 'p50': 1.4190524131256752, 'p75': 2.5554024110222175, 'p90': 4.2200732894330235}

