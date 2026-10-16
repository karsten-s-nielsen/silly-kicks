# TF-79 ball-contact detection validation

Corpus: 966 matches, providers ['gradientsports', 'idsse', 'skillcorner'].
Reported, not gated (no accuracy threshold; spec sec. 8, amended).

SPADL event timestamps LAG the physical ball contact, so a tight symmetric window measures
event-feed misalignment, not detector quality. Hence the characterisation below, not one number.

## Per provider

Authoritative read: each provider is scored ONLY on its own matches, so these blocks are not
diluted by the provider mix. SkillCorner (the primary production provider) is listed first.

### skillcorner
- contacts: 3851841 (resolved 3136240, events 1048765)
- contacts/min: 43.2550547729072
- unresolved rate: 0.18578155224995008
- signed event->contact offset (s): {'p10': -0.28999999999973625, 'p25': -0.10000000000000142, 'p50': 0.0, 'p75': 0.1999999999998181, 'p90': 1.099999999999909}
- agreement vs tolerance: {'agree_within_0.2s': 0.6326784360652767, 'agree_within_0.5s': 0.7897851282222423, 'agree_within_1.0s': 0.844921407560321}
- confidence calibration (event-match rate by bucket): {'low': {'n': 73763, 'event_match_rate': 0.04618846847335385}, 'mid': {'n': 2559760, 'event_match_rate': 0.3077608838328593}, 'high': {'n': 502717, 'event_match_rate': 0.4015559449948977}}
- velocity-change |dv| (m/s): {'p10': 0.41228761854588786, 'p25': 0.7616398232451864, 'p50': 1.421248574721416, 'p75': 2.545507806321914, 'p90': 4.143779442009596}

### gradientsports
- contacts: 387730 (resolved 272759, events 90053)
- contacts/min: 38.56179703762493
- unresolved rate: 0.2965233538802775
- signed event->contact offset (s): {'p10': -10.433178699999964, 'p25': -0.02076199999982009, 'p50': 0.3593275000000631, 'p75': 0.8596452499999998, 'p90': 11.583722900000021}
- agreement vs tolerance: {'agree_within_0.2s': 0.22140295159517173, 'agree_within_0.5s': 0.4181093356134721, 'agree_within_1.0s': 0.6373135820017101}
- confidence calibration (event-match rate by bucket): {'low': {'n': 5175, 'event_match_rate': 0.1267632850241546}, 'mid': {'n': 233896, 'event_match_rate': 0.18014416663816396}, 'high': {'n': 33688, 'event_match_rate': 0.21461648064592734}}
- velocity-change |dv| (m/s): {'p10': 0.5402739114378919, 'p25': 0.9477800667747753, 'p50': 1.6209461737803705, 'p75': 2.5519346117503603, 'p90': 3.794037355154133}

### idsse
- contacts: 25739 (resolved 21974, events 8581)
- contacts/min: 38.507802097515
- unresolved rate: 0.1462760791017522
- signed event->contact offset (s): {'p10': -2.3917999999999693, 'p25': -0.8369999999999749, 'p50': -0.10000000000007958, 'p75': 0.4189999999999987, 'p90': 1.2637000000000624}
- agreement vs tolerance: {'agree_within_0.2s': 0.22712970516256847, 'agree_within_0.5s': 0.4452860971914695, 'agree_within_1.0s': 0.6488754224449365}
- confidence calibration (event-match rate by bucket): {'low': {'n': 123, 'event_match_rate': 0.11382113821138211}, 'mid': {'n': 8688, 'event_match_rate': 0.18945672191528545}, 'high': {'n': 13163, 'event_match_rate': 0.23474891742004103}}
- velocity-change |dv| (m/s): {'p10': 0.7132224704555141, 'p25': 1.7497236901664297, 'p50': 4.883378070723005, 'p75': 10.047275301838472, 'p90': 16.45453337684113}

## Pooled across all providers

_Provider-mix dependent: dominated by whichever provider contributes the most matches (here SkillCorner). This is NOT a detector-quality score -- read the per-provider blocks above._

### overall (pooled)
- contacts: 4265310 (resolved 3430973, events 1147399)
- contacts/min: 42.75027993635132
- unresolved rate: 0.19560993222063577
- signed event->contact offset (s): {'p10': -0.29999999999972715, 'p25': -0.09999999999999964, 'p50': 0.0, 'p75': 0.2000000000007276, 'p90': 1.2000000000002728}
- agreement vs tolerance: {'agree_within_0.2s': 0.597366739904776, 'agree_within_0.5s': 0.7580379623827457, 'agree_within_1.0s': 0.8271612577664788}
- confidence calibration (event-match rate by bucket): {'low': {'n': 79061, 'event_match_rate': 0.05156777677995472}, 'mid': {'n': 2802344, 'event_match_rate': 0.2967426554341651}, 'high': {'n': 549568, 'event_match_rate': 0.38610144695469895}}
- velocity-change |dv| (m/s): {'p10': 0.41232462601712727, 'p25': 0.7810550545203971, 'p50': 1.4418789197394828, 'p75': 2.556634515447175, 'p90': 4.176050111933218}

