# Form-C search output (at `2901fa5c`, tree `527bf49c`)

Read-only array origins, then the consumers of the two object ones (`PitchControlSurface.player_ids` / `player_team_ids`).

```text
$ rg -n "writeable = False|setflags\(write=False|broadcast_to\(|frombuffer\(" silly_kicks scripts
silly_kicks/coordination/_kernels/_surrogates.py:354:    lengths = np.broadcast_to(ns, start_b.shape)
silly_kicks/coordination/_kernels/_cluster.py:309:    a_rows = np.broadcast_to(a, b.shape)
silly_kicks/coordination/_compute.py:909:            da_rows = np.ascontiguousarray(np.broadcast_to(da, db.shape))
silly_kicks/coordination/_compute.py:1665:    a.flags.writeable = False
silly_kicks/tracking/preprocess/_butterworth.py:64:    corrupt every other caller -- ``flags.writeable = False`` makes that fail loud instead. It is deliberately
silly_kicks/tracking/preprocess/_butterworth.py:76:    sos.flags.writeable = False
silly_kicks/tracking/preprocess/_butterworth.py:94:    zi.flags.writeable = False
silly_kicks/tracking/_ghost_gk.py:254:        self.grid_x.flags.writeable = False
silly_kicks/tracking/_ghost_gk.py:255:        self.grid_y.flags.writeable = False
silly_kicks/tracking/_ghost_gk.py:256:        self.probabilities.flags.writeable = False
silly_kicks/tracking/pitch_control/_surface.py:55:        self.grid_x.flags.writeable = False
silly_kicks/tracking/pitch_control/_surface.py:56:        self.grid_y.flags.writeable = False
silly_kicks/tracking/pitch_control/_surface.py:57:        self.surface.flags.writeable = False
silly_kicks/tracking/pitch_control/_surface.py:59:            self.per_player_influence.flags.writeable = False
silly_kicks/tracking/pitch_control/_surface.py:61:            self.player_ids.flags.writeable = False
silly_kicks/tracking/pitch_control/_surface.py:63:            self.player_team_ids.flags.writeable = False
silly_kicks/tracking/pitch_control/_grids.py:31:        arr.setflags(write=False)
silly_kicks/tracking/_gk_influence.py:98:        self.points.flags.writeable = False

$ rg -n "\.player_ids\b|\.player_team_ids\b" silly_kicks scripts
silly_kicks/restdefense/_danger.py:87:    if surface.per_player_influence is None or surface.player_ids is None or surface.player_team_ids is None:
silly_kicks/restdefense/_danger.py:93:    team_mask = ids_match(surface.player_team_ids, team_id).to_numpy()
silly_kicks/tracking/pitch_control/_surface.py:60:        if self.player_ids is not None:
silly_kicks/tracking/pitch_control/_surface.py:61:            self.player_ids.flags.writeable = False
silly_kicks/tracking/pitch_control/_surface.py:62:        if self.player_team_ids is not None:
silly_kicks/tracking/pitch_control/_surface.py:63:            self.player_team_ids.flags.writeable = False
silly_kicks/tracking/pitch_control/_surface.py:161:        if self.per_player_influence is None or self.player_ids is None:
silly_kicks/tracking/pitch_control/_surface.py:166:        idx = np.where(ids_match(self.player_ids, player_id).to_numpy())[0]
silly_kicks/tracking/pitch_control/_surface.py:168:            raise ValueError(f"player_id={player_id!r} not found in player_ids {self.player_ids.tolist()}")
silly_kicks/tracking/pitch_control/_surface.py:173:        if self.player_team_ids is not None:
silly_kicks/tracking/pitch_control/_surface.py:174:            team_id = self.player_team_ids[idx[0]]
silly_kicks/tracking/pitch_control/_surface.py:175:            team_mask = self.player_team_ids == team_id
silly_kicks/tracking/pitch_control/_surface.py:194:        if self.per_player_influence is None or self.player_ids is None:
silly_kicks/tracking/pitch_control/_surface.py:198:        idx = np.where(ids_match(self.player_ids, player_id).to_numpy())[0]
silly_kicks/tracking/pitch_control/_surface.py:200:            raise ValueError(f"player_id={player_id!r} not found in player_ids {self.player_ids.tolist()}")
silly_kicks/tracking/_gk_influence.py:390:        surface.player_ids is not None
silly_kicks/tracking/_gk_influence.py:391:        and surface.player_team_ids is not None
silly_kicks/tracking/_gk_influence.py:396:        team_mask_arr = surface.player_team_ids == defending_team_id
silly_kicks/tracking/_run_values.py:664:        pidx = _safe_index_of(pc.player_ids, out.at[ridx, "player_id"])
silly_kicks/tracking/_space_creation.py:389:    p_ids = np.asarray(baseline_surface.player_ids)  # type: ignore[union-attr]
silly_kicks/tracking/_space_creation.py:390:    p_teams = np.asarray(baseline_surface.player_team_ids)  # type: ignore[union-attr]
```
